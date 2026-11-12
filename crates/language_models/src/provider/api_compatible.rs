use std::{sync::Arc, time::Duration};

use ::util::ResultExt;
use anyhow::{Context as _, Result};
use collections::HashSet;
use convert_case::{Case, Casing};
use credentials_provider::CredentialsProvider;
use futures::AsyncReadExt;
use gpui::{App, AppContext as _, Context, Entity, SharedString, Task, TaskExt, Window};
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, HttpRequestExt as _, Method, Request as HttpRequest,
    RequestBuilderExt,
};
use language_model::{ApiKeyState, AuthenticateError, EnvVar};
use serde::Deserialize;
use settings::SettingsStore;
use ui::{ElevationIndex, Tooltip, prelude::*};
use ui_input::InputField;

pub trait ApiCompatibleProviderSettings: Clone + Default + PartialEq + 'static {
    fn api_url(&self) -> &str;
}

/// Which compatible-provider API convention a model list request should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatibleProviderFlavor {
    OpenAi,
    Anthropic,
}

/// A model advertised by a compatible provider's model-list endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchedModel {
    pub name: String,
    pub display_name: Option<String>,
}

/// Upper bound for a model list request. The dialog disables its fetch button while
/// the request is in flight, so hanging here would strand the user.
const MODEL_LIST_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Fetches the model list advertised by a compatible provider's API.
///
/// Model-list endpoints usually expose only identifiers and display names, so
/// callers start fetched models from their existing defaults and let the user
/// refine capabilities before saving.
pub async fn fetch_models(
    client: Arc<dyn HttpClient>,
    flavor: CompatibleProviderFlavor,
    api_url: &str,
    api_key: Option<&str>,
    custom_headers: &CustomHeaders,
) -> Result<Vec<FetchedModel>> {
    let (uri, api_key_header) = match flavor {
        CompatibleProviderFlavor::OpenAi => (
            format!("{}/models", api_url.trim_end_matches('/')),
            "Authorization",
        ),
        CompatibleProviderFlavor::Anthropic => (
            format!("{}/v1/models?limit=1000", api_url.trim_end_matches('/')),
            "X-Api-Key",
        ),
    };

    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(uri.clone())
        // A model list endpoint that accepts the connection but never replies would
        // otherwise leave the dialog stuck on "Fetching…" with no way to retry.
        .timeout(MODEL_LIST_REQUEST_TIMEOUT);

    // Custom headers are appended to the request, so a custom `Authorization`
    // would sit next to the managed one. Drop the headers this request owns and
    // let the flavor-specific values below provide them.
    let reserved_header_names: &[&str] = match flavor {
        CompatibleProviderFlavor::OpenAi => &["authorization", "content-type", "accept"],
        CompatibleProviderFlavor::Anthropic => &[
            "authorization",
            "content-type",
            "accept",
            "x-api-key",
            "anthropic-version",
            "anthropic-beta",
        ],
    };
    let custom_headers = CustomHeaders::new(
        custom_headers
            .iter()
            .filter(|(name, _)| !reserved_header_names.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    );
    let mut request = request
        .header("Accept", "application/json")
        .extra_headers(&custom_headers);
    if flavor == CompatibleProviderFlavor::Anthropic {
        request = request.header("Anthropic-Version", "2023-06-01");
    }
    if let Some(api_key) = api_key.filter(|api_key| !api_key.trim().is_empty()) {
        let value = match flavor {
            CompatibleProviderFlavor::OpenAi => format!("Bearer {}", api_key.trim()),
            CompatibleProviderFlavor::Anthropic => api_key.trim().to_string(),
        };
        request = request.header(api_key_header, value);
    }

    let request = request
        .body(AsyncBody::default())
        .with_context(|| format!("failed to build a model list request for {uri}"))?;
    let host = request.uri().host().unwrap_or(api_url).to_owned();
    let mut response = client
        .send(request)
        .await
        .with_context(|| format!("failed to reach {host}"))?;
    let status = response.status();
    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .with_context(|| format!("failed to read the model list response from {host}"))?;

    if !status.is_success() {
        let detail: String = body.trim().chars().take(300).collect();
        if detail.is_empty() {
            anyhow::bail!("{host} returned {status}");
        } else {
            anyhow::bail!("{host} returned {status}: {detail}");
        }
    }

    parse_models(&body).with_context(|| format!("failed to parse the model list from {host}"))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ModelListResponse {
    Data { data: Vec<ModelListEntry> },
    Models { models: Vec<ModelListEntry> },
    List(Vec<ModelListEntry>),
}

#[derive(Deserialize)]
struct ModelListEntry {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    display_name: Option<String>,
}

fn parse_models(body: &str) -> Result<Vec<FetchedModel>> {
    let entries = match serde_json::from_str::<ModelListResponse>(body)? {
        ModelListResponse::Data { data } => data,
        ModelListResponse::Models { models } => models,
        ModelListResponse::List(models) => models,
    };

    let mut seen = HashSet::default();
    Ok(entries
        .into_iter()
        .filter_map(|entry| {
            let name = entry.id.or(entry.name)?.trim().to_string();
            if name.is_empty() || !seen.insert(name.clone()) {
                return None;
            }
            let display_name = entry
                .display_name
                .map(|display_name| display_name.trim().to_string())
                .filter(|display_name| !display_name.is_empty() && display_name != &name);
            Some(FetchedModel { name, display_name })
        })
        .collect())
}

pub struct ApiCompatibleProviderState<S: ApiCompatibleProviderSettings> {
    id: Arc<str>,
    pub api_key_state: ApiKeyState,
    pub settings: S,
    credentials_provider: Arc<dyn CredentialsProvider>,
}

impl<S: ApiCompatibleProviderSettings> ApiCompatibleProviderState<S> {
    pub fn new(
        id: Arc<str>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        resolve_settings: for<'a> fn(&'a str, &'a App) -> Option<&'a S>,
        cx: &mut App,
    ) -> Entity<Self> {
        let api_key_env_var_name: SharedString =
            format!("{}_API_KEY", id).to_case(Case::UpperSnake).into();
        cx.new(|cx| {
            cx.observe_global::<SettingsStore>(move |this: &mut Self, cx| {
                let Some(settings) = resolve_settings(&this.id, cx).cloned() else {
                    return;
                };
                this.update_settings(settings, cx);
            })
            .detach();

            let settings = resolve_settings(&id, cx).cloned().unwrap_or_default();
            let api_key_scope = SharedString::new(id.as_ref());
            Self {
                id,
                api_key_state: ApiKeyState::new_with_scope(
                    SharedString::new(settings.api_url()),
                    Some(api_key_scope),
                    EnvVar::new(api_key_env_var_name),
                ),
                settings,
                credentials_provider,
            }
        })
    }

    pub fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key()
    }

    pub fn set_api_key(
        &mut self,
        api_key: Option<String>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let api_url = SharedString::new(self.settings.api_url());
        self.api_key_state.store(
            api_url,
            api_key,
            |this| &mut this.api_key_state,
            self.credentials_provider.clone(),
            cx,
        )
    }

    pub fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let api_url = SharedString::new(self.settings.api_url());
        self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            self.credentials_provider.clone(),
            cx,
        )
    }

    pub fn update_settings(&mut self, settings: S, cx: &mut Context<Self>) {
        if self.settings != settings {
            let api_url = SharedString::new(settings.api_url());
            self.api_key_state.handle_url_change(
                api_url,
                |this| &mut this.api_key_state,
                self.credentials_provider.clone(),
                cx,
            );
            self.settings = settings;
            cx.notify();
        }
    }
}

pub struct ApiCompatibleProviderConfigurationView<S: ApiCompatibleProviderSettings> {
    api_key_editor: Entity<InputField>,
    state: Entity<ApiCompatibleProviderState<S>>,
    provider_name: &'static str,
    load_credentials_task: Option<Task<()>>,
}

impl<S: ApiCompatibleProviderSettings> ApiCompatibleProviderConfigurationView<S> {
    pub fn new(
        state: Entity<ApiCompatibleProviderState<S>>,
        provider_name: &'static str,
        placeholder_text: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let api_key_editor = cx.new(|cx| InputField::new(window, cx, placeholder_text));

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            async move |this, cx| {
                let task = state.update(cx, |state, cx| state.authenticate(cx));
                match task.await {
                    Ok(()) | Err(AuthenticateError::CredentialsNotFound) => {}
                    Err(error) => {
                        log::error!(
                            "Failed to load {provider_name}-compatible provider API credentials: {error}"
                        );
                    }
                }
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor,
            state,
            provider_name,
            load_credentials_task,
        }
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        // url changes can cause the editor to be displayed again
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn remove_provider(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let id = self.state.read(cx).id.clone();
        let fs = <dyn fs::Fs>::global(cx);
        settings::update_settings_file(fs, cx, move |settings, _| {
            let Some(language_models) = settings.language_models.as_mut() else {
                return;
            };

            if let Some(providers) = language_models.openai_compatible.as_mut() {
                providers.remove(id.as_ref());
            }
            if let Some(providers) = language_models.anthropic_compatible.as_mut() {
                providers.remove(id.as_ref());
            }
        });
    }

    fn should_render_editor(&self, cx: &Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl<S: ApiCompatibleProviderSettings> Render for ApiCompatibleProviderConfigurationView<S> {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let env_var_set = state.api_key_state.is_from_env_var();
        let env_var_name = state.api_key_state.env_var_name();
        let provider_name = self.provider_name;

        let api_key_section = if self.should_render_editor(cx) {
            v_flex()
                .on_action(cx.listener(Self::save_api_key))
                .child(Label::new(format!(
                    "To use Zed's agent with an {provider_name}-compatible provider, you need to add an API key."
                )))
                .child(
                    div()
                        .pt(DynamicSpacing::Base04.rems(cx))
                        .child(self.api_key_editor.clone()),
                )
                .child(
                    Label::new(format!(
                        "You can also set the {env_var_name} environment variable and restart Zed.",
                    ))
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
                .into_any()
        } else {
            h_flex()
                .mt_1()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_2()
                .border_color(cx.theme().colors().border)
                .bg(cx.theme().colors().background)
                .child(
                    h_flex()
                        .flex_1()
                        .min_w_0()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(
                            div().w_full().overflow_x_hidden().text_ellipsis().child(Label::new(
                                if env_var_set {
                                    format!("API key set in {env_var_name} environment variable")
                                } else {
                                    format!("API key configured for {}", state.settings.api_url())
                                },
                            )),
                        ),
                )
                .child(
                    h_flex().flex_shrink_0().child(
                        Button::new("reset-api-key", "Reset API Key")
                            .label_size(LabelSize::Small)
                            .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                            .layer(ElevationIndex::ModalSurface)
                            .when(env_var_set, |this| {
                                this.tooltip(Tooltip::text(format!(
                                    "To reset your API key, unset the {env_var_name} environment variable.",
                                )))
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.reset_api_key(window, cx)
                            })),
                    ),
                )
                .into_any()
        };

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials…")).into_any()
        } else {
            v_flex()
                .size_full()
                .gap_4()
                .child(api_key_section)
                .child(
                    h_flex().w_full().justify_end().child(
                        Button::new("remove-compatible-provider", "Remove Provider")
                            .style(ButtonStyle::OutlinedGhost)
                            .label_size(LabelSize::Small)
                            .start_icon(
                                Icon::new(IconName::Trash)
                                    .size(IconSize::Small)
                                    .color(Color::Muted),
                            )
                            .on_click(cx.listener(|this, _event, window, cx| {
                                this.remove_provider(window, cx)
                            })),
                    ),
                )
                .into_any()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_open_ai_style_model_lists() {
        let models = parse_models(r#"{"data":[{"id":"gpt-4o"},{"id":"gpt-4o-mini"}]}"#)
            .expect("parse OpenAI model list");
        assert_eq!(
            models,
            vec![
                FetchedModel {
                    name: "gpt-4o".into(),
                    display_name: None,
                },
                FetchedModel {
                    name: "gpt-4o-mini".into(),
                    display_name: None,
                },
            ]
        );
    }

    #[test]
    fn parses_anthropic_style_model_lists_and_removes_duplicates() {
        let models = parse_models(
            r#"{"data":[{"id":"claude-3","display_name":"Claude 3"},{"id":"claude-3","display_name":"Claude 3"},{"id":"claude-4","display_name":"claude-4"}]}"#,
        )
        .expect("parse Anthropic model list");
        assert_eq!(
            models,
            vec![
                FetchedModel {
                    name: "claude-3".into(),
                    display_name: Some("Claude 3".into()),
                },
                FetchedModel {
                    name: "claude-4".into(),
                    display_name: None,
                },
            ]
        );
    }

    #[test]
    fn parses_bare_model_arrays_and_name_fields() {
        let models = parse_models(r#"[{"name":"local-model"}]"#).expect("parse bare model array");
        assert_eq!(
            models,
            vec![FetchedModel {
                name: "local-model".into(),
                display_name: None,
            }]
        );
    }
}
