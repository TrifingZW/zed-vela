//! Zed Vela: an LLM provider configuration dialog for the native agent.
//!
//! Upstream moved provider configuration into the settings UI's LLM Providers
//! page. Zed Vela keeps the agent-panel model menu's **Configure** action
//! pointing at this dialog instead, so providers can still be configured
//! without leaving the agent panel. The dialog mirrors that settings page,
//! including its sub-pages, using in-dialog navigation instead of a settings
//! window.

use std::{collections::HashSet, sync::Arc};

use collections::HashMap;
use editor::Editor;
use gpui::{
    AnyElement, AnyView, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, ScrollHandle,
    Subscription, prelude::*,
};
use language_model::{
    ApiKeyConfiguration, CreateProviderSettingsView, IconOrSvg, InlineDescription,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelRegistry, ModelMode,
    ProviderSettingsView,
};
use settings::{
    AnthropicCompatibleAvailableModel, AnthropicCompatibleModelCapabilities,
    AnthropicCompatibleSettingsContent, OpenAiCompatibleAvailableModel,
    OpenAiCompatibleModelCapabilities, OpenAiCompatibleSettingsContent, OpenAiReasoningEffort,
    SettingsStore,
};
use ui::{
    ButtonLink, Checkbox, ConfiguredApiCard, ContextMenu, Divider, DividerColor, DropdownMenu,
    DropdownStyle, IconPosition, KeyBinding, Modal, ModalFooter, ModalHeader, PopoverMenu,
    ToggleState, Tooltip, prelude::*,
};
use util::ResultExt as _;
use workspace::{DismissDecision, ModalView, Workspace};

enum Mode {
    /// The provider list, mirroring the LLM Providers settings page.
    List,
    /// A provider's own configuration, which the settings page opens as a sub-page.
    ProviderConfig {
        provider_id: LanguageModelProviderId,
    },
    /// The "Compatible APIs" form, which the settings page opens as a sub-page.
    AddProvider(Box<LlmProviderForm>),
}

/// A snapshot of the mode used while rendering, so building the view can borrow
/// `self` mutably for listeners.
enum RenderMode {
    List,
    ProviderConfig(LanguageModelProviderId),
    AddProvider(CompatibleProviderKind),
}

pub(crate) struct ConfigureModelsDialog {
    focus_handle: FocusHandle,
    scroll_handle: ScrollHandle,
    mode: Mode,
    /// Providers whose API key section has an editor currently mounted. API keys
    /// are stored in the system keychain, so the input stays alive until the
    /// provider reports a key, which re-creates the row.
    api_key_editors: HashMap<LanguageModelProviderId, ApiKeyEditor>,
    /// Provider configuration views, created once per provider so in-progress
    /// edits survive re-renders.
    provider_views: HashMap<LanguageModelProviderId, AnyView>,
    /// Whether the user asked to close the dialog.
    ///
    /// Clicks outside the dialog and focus changes also route through
    /// `ModalView::on_before_dismiss`, so dismissal has to be gated on this flag
    /// rather than on the dismissal source.
    dismissed: bool,
    _subscriptions: Vec<Subscription>,
}

struct ApiKeyEditor {
    provider: Arc<dyn LanguageModelProvider>,
    editor: Entity<Editor>,
    _subscription: Subscription,
}

/// Stores the API key currently typed into `editor`, if any.
fn submit_api_key(
    provider: &Arc<dyn LanguageModelProvider>,
    editor: &Entity<Editor>,
    cx: &mut App,
) -> bool {
    let api_key = editor.read(cx).text(cx).trim().to_string();
    if api_key.is_empty() {
        return false;
    }

    provider
        .set_api_key(Some(api_key), cx)
        .detach_and_log_err(cx);
    true
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompatibleProviderKind {
    OpenAi,
    Anthropic,
}

impl CompatibleProviderKind {
    fn label(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI",
            Self::Anthropic => "Anthropic",
        }
    }

    fn default_api_url(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::OpenAi => "This provider will use an OpenAI-compatible API.",
            Self::Anthropic => "This provider will use an Anthropic Messages-compatible API.",
        }
    }
}

struct LlmProviderForm {
    kind: CompatibleProviderKind,
    provider_name: Entity<Editor>,
    api_url: Entity<Editor>,
    api_key: Entity<Editor>,
    custom_headers: Vec<HeaderInput>,
    models: Vec<ModelInput>,
    error: Option<SharedString>,
}

struct HeaderInput {
    name: Entity<Editor>,
    value: Entity<Editor>,
}

impl LlmProviderForm {
    fn new(kind: CompatibleProviderKind, window: &mut Window, cx: &mut App) -> Self {
        Self {
            kind,
            provider_name: new_input(kind.label(), None, false, window, cx),
            api_url: new_input(kind.default_api_url(), None, false, window, cx),
            api_key: new_input(
                "000000000000000000000000000000000000000000000000",
                None,
                true,
                window,
                cx,
            ),
            custom_headers: Vec::new(),
            models: vec![ModelInput::new(window, cx)],
            error: None,
        }
    }
}

struct ModelInput {
    name: Entity<Editor>,
    display_name: Entity<Editor>,
    max_completion_tokens: Entity<Editor>,
    max_output_tokens: Entity<Editor>,
    max_tokens: Entity<Editor>,
    reasoning_effort: OpenAiReasoningEffort,
    tool_override: Entity<Editor>,
    default_temperature: Entity<Editor>,
    extra_beta_headers: Entity<Editor>,
    mode: ModelModeInput,
    thinking_budget_tokens: Entity<Editor>,
    supports_tools: ToggleState,
    supports_images: ToggleState,
    supports_parallel_tool_calls: ToggleState,
    supports_prompt_cache_key: ToggleState,
    supports_chat_completions: ToggleState,
    supports_thinking: ToggleState,
    interleaved_reasoning: ToggleState,
    max_tokens_parameter: ToggleState,
    supports_prompt_caching: ToggleState,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelModeInput {
    Default,
    Thinking,
    Adaptive,
}

impl ModelModeInput {
    const SELECTABLE: [Self; 3] = [Self::Default, Self::Thinking, Self::Adaptive];

    fn label(self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::Thinking => "Thinking",
            Self::Adaptive => "Adaptive",
        }
    }
}

impl ModelInput {
    fn new(window: &mut Window, cx: &mut App) -> Self {
        let OpenAiCompatibleModelCapabilities {
            tools,
            images,
            parallel_tool_calls,
            prompt_cache_key,
            chat_completions,
            interleaved_reasoning,
            max_tokens_parameter,
        } = OpenAiCompatibleModelCapabilities::default();

        Self {
            name: new_input(
                "e.g. gpt-5, claude-opus-4, gemini-2.5-pro",
                None,
                false,
                window,
                cx,
            ),
            display_name: new_input("e.g. GPT-5", None, false, window, cx),
            max_completion_tokens: new_input("200000", Some("200000"), false, window, cx),
            max_output_tokens: new_input("Max Output Tokens", Some("32000"), false, window, cx),
            max_tokens: new_input("Max Tokens", Some("200000"), false, window, cx),
            reasoning_effort: OpenAiReasoningEffort::Medium,
            tool_override: new_input("Model name to use for tool calls", None, false, window, cx),
            default_temperature: new_input("1.0", None, false, window, cx),
            extra_beta_headers: new_input("header-1, header-2", None, false, window, cx),
            mode: ModelModeInput::Default,
            thinking_budget_tokens: new_input("4096", None, false, window, cx),
            supports_tools: tools.into(),
            supports_images: images.into(),
            supports_parallel_tool_calls: parallel_tool_calls.into(),
            supports_prompt_cache_key: prompt_cache_key.into(),
            supports_chat_completions: chat_completions.into(),
            supports_thinking: ToggleState::Unselected,
            interleaved_reasoning: interleaved_reasoning.into(),
            max_tokens_parameter: max_tokens_parameter.into(),
            supports_prompt_caching: ToggleState::Unselected,
        }
    }
}

fn new_input(
    placeholder: &str,
    initial: Option<&str>,
    masked: bool,
    window: &mut Window,
    cx: &mut App,
) -> Entity<Editor> {
    let placeholder = placeholder.to_string();
    let initial = initial.map(str::to_string);
    cx.new(|cx| {
        let mut editor = Editor::single_line(window, cx);
        editor.set_placeholder_text(placeholder.as_str(), window, cx);
        editor.set_masked(masked, cx);
        if let Some(text) = initial {
            editor.set_text(text, window, cx);
        }
        editor
    })
}

impl ConfigureModelsDialog {
    pub fn register(
        workspace: &mut Workspace,
        _window: Option<&mut Window>,
        _cx: &mut Context<Workspace>,
    ) {
        workspace.register_action(
            |workspace, _: &zed_actions::agent::ConfigureModels, window, cx| {
                workspace.toggle_modal(window, cx, |window, cx| Self::new(window, cx));
            },
        );
    }

    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();

        // Provider authentication state changes are published as registry events
        // (not `cx.notify` notifications), while API keys and provider settings land
        // in the settings store. Re-render on both so the dialog never shows stale
        // provider state.
        let registry = LanguageModelRegistry::global(cx);
        let registry_subscription = cx.subscribe(
            &registry,
            |_, _, event: &language_model::Event, cx| match event {
                language_model::Event::ProviderStateChanged(_)
                | language_model::Event::AddedProvider(_)
                | language_model::Event::RemovedProvider(_)
                | language_model::Event::ProvidersChanged => cx.notify(),
                _ => {}
            },
        );
        let settings_subscription = cx.observe_global_in::<SettingsStore>(window, |_, _, cx| {
            cx.notify();
        });

        Self {
            focus_handle,
            scroll_handle: ScrollHandle::new(),
            mode: Mode::List,
            api_key_editors: HashMap::default(),
            provider_views: HashMap::default(),
            dismissed: false,
            _subscriptions: vec![registry_subscription, settings_subscription],
        }
    }

    fn render_mode(&self) -> RenderMode {
        match &self.mode {
            Mode::List => RenderMode::List,
            Mode::ProviderConfig { provider_id } => RenderMode::ProviderConfig(provider_id.clone()),
            Mode::AddProvider(form) => RenderMode::AddProvider(form.kind),
        }
    }

    fn set_mode(&mut self, mode: Mode, window: &mut Window, cx: &mut Context<Self>) {
        self.mode = mode;
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    fn dismiss(&mut self, cx: &mut Context<Self>) {
        // Pressing Escape hides the dialog before the focused editor can blur, so
        // commit any key still sitting in an input or it would be lost.
        for api_key_editor in self.api_key_editors.values() {
            submit_api_key(&api_key_editor.provider, &api_key_editor.editor, cx);
        }
        self.dismissed = true;
        cx.emit(DismissEvent);
    }

    fn go_back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.set_mode(Mode::List, window, cx);
    }

    fn open_provider_config(
        &mut self,
        provider_id: LanguageModelProviderId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_mode(Mode::ProviderConfig { provider_id }, window, cx);
    }

    fn open_add_provider_form(
        &mut self,
        kind: CompatibleProviderKind,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let form = Box::new(LlmProviderForm::new(kind, window, cx));
        self.set_mode(Mode::AddProvider(form), window, cx);
    }

    fn configuration_view(
        &mut self,
        provider_id: &LanguageModelProviderId,
        create_view: CreateProviderSettingsView,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyView {
        if let Some(view) = self.provider_views.get(provider_id) {
            return view.clone();
        }

        let view = create_view(window, cx);
        self.provider_views
            .insert(provider_id.clone(), view.clone());
        view
    }

    // -- Provider list ------------------------------------------------------

    fn render_list(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let providers = LanguageModelRegistry::read_global(cx).visible_providers();

        let sections = providers
            .iter()
            .filter_map(|provider| self.provider_section(provider, window, cx))
            .collect::<Vec<_>>();

        v_flex()
            .w_full()
            .min_w_0()
            .gap_6()
            .when(sections.is_empty(), |this| {
                this.child(
                    Label::new("No LLM providers are available.")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
            })
            .children(sections)
            .into_any_element()
    }

    fn provider_section(
        &mut self,
        provider: &Arc<dyn LanguageModelProvider>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let provider_id = provider.id();
        let provider_name = provider.name().0;
        let icon = provider.icon();

        let body = match provider.settings_view(cx)? {
            ProviderSettingsView::ApiKey(config) => {
                self.render_api_key_section(provider, provider_name.clone(), config, window, cx)
            }
            ProviderSettingsView::Inline(settings) => {
                let view = self.configuration_view(&provider_id, settings.create_view, window, cx);
                Self::render_inline_body(
                    provider_name.clone(),
                    settings.title,
                    settings.description,
                    view,
                )
            }
            ProviderSettingsView::SubPage(settings) => {
                self.render_sub_page_item(provider, settings.description, cx)
            }
        };

        Some(
            v_flex()
                .w_full()
                .min_w_0()
                .gap_1p5()
                .child(Self::render_provider_header(provider_name, icon, cx))
                .child(body)
                .into_any_element(),
        )
    }

    fn render_api_key_section(
        &mut self,
        provider: &Arc<dyn LanguageModelProvider>,
        provider_name: SharedString,
        config: ApiKeyConfiguration,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let provider_id = provider.id();

        if config.has_key {
            // Drop any editor we no longer need once the key is stored.
            self.api_key_editors.remove(&provider_id);
            return Self::render_configured_api_key(provider, config);
        }

        let editor = if let Some(api_key_editor) = self.api_key_editors.get(&provider_id) {
            api_key_editor.editor.clone()
        } else {
            let editor = cx.new(|cx| {
                let mut editor = Editor::single_line(window, cx);
                editor.set_placeholder_text("xxxxxxxxxxxxxxxxxxxx", window, cx);
                editor.set_masked(true, cx);
                editor
            });

            let provider_for_input = provider.clone();
            let subscription =
                cx.subscribe_in(&editor, window, move |_, editor, event, window, cx| {
                    if !matches!(event, editor::EditorEvent::Blurred) {
                        return;
                    }

                    if submit_api_key(&provider_for_input, editor, cx) {
                        editor.update(cx, |editor, cx| editor.set_text("", window, cx));
                    }
                });

            self.api_key_editors.insert(
                provider_id,
                ApiKeyEditor {
                    provider: provider.clone(),
                    editor: editor.clone(),
                    _subscription: subscription,
                },
            );
            editor
        };

        Self::render_api_key_input(provider_name, config, editor, cx)
    }

    fn render_configured_api_key(
        provider: &Arc<dyn LanguageModelProvider>,
        config: ApiKeyConfiguration,
    ) -> AnyElement {
        let is_from_env_var = config.is_from_env_var;
        let env_var_name = config.env_var_name;
        let button_id = format!("reset-api-key-{}", provider.id().0);

        let configured_label = if is_from_env_var {
            "API Key Set in Environment Variable"
        } else {
            "API Key Configured"
        };

        let provider = provider.clone();
        v_flex()
            .w_full()
            .min_w_0()
            .gap_2()
            .child(
                ConfiguredApiCard::new(button_id, configured_label)
                    .button_label("Reset Key")
                    .button_tab_index(0)
                    .disabled(is_from_env_var)
                    .when(is_from_env_var, |this| {
                        this.tooltip_label(format!(
                            "To reset your API key, unset the {env_var_name} environment variable."
                        ))
                    })
                    .on_click(move |_, _, cx| {
                        provider.set_api_key(None, cx).detach_and_log_err(cx);
                    }),
            )
            .into_any_element()
    }

    fn render_api_key_input(
        provider_name: SharedString,
        config: ApiKeyConfiguration,
        editor: Entity<Editor>,
        cx: &App,
    ) -> AnyElement {
        let colors = cx.theme().colors();
        let env_var_name = config.env_var_name;
        let api_key_url = config.api_key_url;
        let focus_handle = editor.focus_handle(cx).tab_index(0).tab_stop(true);

        h_flex()
            .pt_2p5()
            .w_full()
            .min_w_0()
            .gap_4()
            .justify_between()
            .child(
                v_flex()
                    .w_full()
                    .min_w_0()
                    .max_w_1_2()
                    .gap_0p5()
                    .child(Label::new("API Key"))
                    .child(
                        h_flex()
                            .w_full()
                            .min_w_0()
                            .flex_wrap()
                            .gap_0p5()
                            .child(
                                Label::new("Visit the")
                                    .size(LabelSize::Small)
                                    .color(Color::Muted),
                            )
                            .child(
                                ButtonLink::new(format!("{provider_name} dashboard"), api_key_url)
                                    .no_icon(true)
                                    .label_size(LabelSize::Small)
                                    .label_color(Color::Muted),
                            )
                            .child(
                                Label::new("to generate an API key.")
                                    .size(LabelSize::Small)
                                    .color(Color::Muted),
                            ),
                    )
                    .child(
                        Label::new(format!(
                            "Or set the {env_var_name} env var and restart Zed for it to take effect."
                        ))
                        .size(LabelSize::XSmall)
                        .color(Color::Muted),
                    ),
            )
            .child(
                h_flex()
                    .min_w_0()
                    .max_w_1_2()
                    .flex_1()
                    .min_w_64()
                    .h_8()
                    .px_2()
                    .rounded_md()
                    .border_2()
                    .border_color(colors.border)
                    .bg(colors.editor_background)
                    .track_focus(&focus_handle)
                    .focus(|style| style.border_color(colors.border_focused))
                    .child(editor),
            )
            .into_any_element()
    }

    fn render_sub_page_item(
        &mut self,
        provider: &Arc<dyn LanguageModelProvider>,
        description: Option<InlineDescription>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let provider_id = provider.id();
        let provider_name = provider.name().0;

        h_flex()
            .pt_2p5()
            .w_full()
            .min_w_0()
            .gap_4()
            .justify_between()
            .child(
                v_flex()
                    .w_full()
                    .min_w_0()
                    .max_w_1_2()
                    .gap_0p5()
                    .child(Label::new("Configure Provider"))
                    .when_some(description, |this, description| {
                        this.child(Self::render_inline_description(provider_name, description))
                    }),
            )
            .child(
                Button::new(format!("configure-{}", provider_id.0), "Configure")
                    .style(ButtonStyle::OutlinedGhost)
                    .size(ButtonSize::Medium)
                    .end_icon(
                        Icon::new(IconName::ChevronRight)
                            .size(IconSize::Small)
                            .color(Color::Muted),
                    )
                    .tab_index(0isize)
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.open_provider_config(provider_id.clone(), window, cx);
                    })),
            )
            .into_any_element()
    }

    fn render_inline_body(
        provider_name: SharedString,
        title: Option<SharedString>,
        description: Option<InlineDescription>,
        view: AnyView,
    ) -> AnyElement {
        if title.is_none() && description.is_none() {
            return v_flex()
                .pt_1()
                .w_full()
                .min_w_0()
                .child(view)
                .into_any_element();
        }

        h_flex()
            .pt_2p5()
            .w_full()
            .min_w_0()
            .gap_4()
            .justify_between()
            .child(
                v_flex()
                    .w_full()
                    .min_w_0()
                    .max_w_1_2()
                    .when_some(title, |this, title| this.child(Label::new(title)))
                    .when_some(description, |this, description| {
                        this.child(Self::render_inline_description(provider_name, description))
                    }),
            )
            .child(
                h_flex()
                    .min_w_0()
                    .max_w_1_2()
                    .flex_1()
                    .justify_end()
                    .child(view),
            )
            .into_any_element()
    }

    fn render_inline_description(
        provider_name: SharedString,
        description: InlineDescription,
    ) -> AnyElement {
        match description {
            InlineDescription::ApiKeyUrl(url) => h_flex()
                .gap_0p5()
                .child(
                    Label::new("To find an API key, visit the")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .child(
                    ButtonLink::new(format!("{provider_name} dashboard."), url)
                        .label_size(LabelSize::Small),
                )
                .into_any_element(),
            InlineDescription::Text(text) => Label::new(text)
                .size(LabelSize::Small)
                .color(Color::Muted)
                .into_any_element(),
        }
    }

    fn render_provider_header(
        provider_name: SharedString,
        icon: IconOrSvg,
        cx: &App,
    ) -> impl IntoElement {
        let icon = match icon {
            IconOrSvg::Svg(path) => Icon::from_external_svg(path),
            IconOrSvg::Icon(name) => Icon::new(name),
        }
        .color(Color::Muted);

        v_flex()
            .w_full()
            .gap_1p5()
            .child(
                h_flex().gap_1p5().child(icon).child(
                    Label::new(provider_name)
                        .size(LabelSize::Small)
                        .color(Color::Muted)
                        .buffer_font(cx),
                ),
            )
            .child(Divider::horizontal().color(DividerColor::BorderFaded))
    }

    fn render_add_provider_button(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let focus_handle = self.focus_handle.clone().tab_index(0).tab_stop(true);
        let dialog = cx.entity().downgrade();

        PopoverMenu::new("add-llm-provider-popover")
            .trigger(
                Button::new("add-llm-provider", "Add Provider")
                    .style(ButtonStyle::Outlined)
                    .track_focus(&focus_handle)
                    .label_size(LabelSize::Small)
                    .start_icon(
                        Icon::new(IconName::Plus)
                            .size(IconSize::Small)
                            .color(Color::Muted),
                    ),
            )
            .anchor(gpui::Anchor::BottomLeft)
            .offset(gpui::Point {
                x: px(0.0),
                y: px(2.0),
            })
            .menu(move |window, cx| {
                let dialog = dialog.clone();
                Some(ContextMenu::build(window, cx, move |menu, _window, _cx| {
                    menu.header("Compatible APIs")
                        .entry("OpenAI", None, {
                            let dialog = dialog.clone();
                            move |window, cx| {
                                dialog
                                    .update(cx, |this, cx| {
                                        this.open_add_provider_form(
                                            CompatibleProviderKind::OpenAi,
                                            window,
                                            cx,
                                        );
                                    })
                                    .log_err();
                            }
                        })
                        .entry("Anthropic", None, {
                            let dialog = dialog.clone();
                            move |window, cx| {
                                dialog
                                    .update(cx, |this, cx| {
                                        this.open_add_provider_form(
                                            CompatibleProviderKind::Anthropic,
                                            window,
                                            cx,
                                        );
                                    })
                                    .log_err();
                            }
                        })
                }))
            })
    }

    // -- Provider configuration sub-page ------------------------------------

    fn render_provider_config(
        &mut self,
        provider_id: &LanguageModelProviderId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(provider) = LanguageModelRegistry::read_global(cx).provider(provider_id) else {
            return Label::new("This provider is no longer available.")
                .size(LabelSize::Small)
                .color(Color::Muted)
                .into_any_element();
        };

        let Some(create_view) =
            provider
                .settings_view(cx)
                .and_then(|settings_view| match settings_view {
                    ProviderSettingsView::Inline(settings) => Some(settings.create_view),
                    ProviderSettingsView::SubPage(settings) => Some(settings.create_view),
                    ProviderSettingsView::ApiKey(_) => None,
                })
        else {
            return Label::new("This provider has no configuration options.")
                .size(LabelSize::Small)
                .color(Color::Muted)
                .into_any_element();
        };

        let view = self.configuration_view(provider_id, create_view, window, cx);

        v_flex().w_full().min_w_0().child(view).into_any_element()
    }

    // -- Add provider form --------------------------------------------------

    fn render_add_provider_form(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Mode::AddProvider(form) = &self.mode else {
            return div().into_any_element();
        };

        v_flex()
            .w_full()
            .min_w_0()
            .gap_4()
            .child(Label::new(form.kind.description()))
            .child(Divider::horizontal().flex_shrink_0())
            .child(render_form_field(
                "Provider Name",
                "A unique name used to identify this provider.",
                &form.provider_name,
                true,
                cx,
            ))
            .child(render_form_field(
                "API URL",
                "The base URL for the compatible API.",
                &form.api_url,
                true,
                cx,
            ))
            .child(render_form_field(
                "API Key",
                "Stored in the system keychain, not in settings.json.",
                &form.api_key,
                true,
                cx,
            ))
            .child(render_custom_headers(form, cx))
            .child(render_models_section(form, window, cx))
            .when_some(form.error.clone(), |this, error| {
                this.child(render_form_error(error))
            })
            .into_any_element()
    }

    fn save_add_provider_form(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let values = {
            let Mode::AddProvider(form) = &self.mode else {
                return;
            };
            LlmProviderFormValues {
                kind: form.kind,
                provider_name: form.provider_name.read(cx).text(cx),
                api_url: form.api_url.read(cx).text(cx),
                api_key: form.api_key.read(cx).text(cx),
                custom_headers: form
                    .custom_headers
                    .iter()
                    .map(|header| {
                        (
                            header.name.read(cx).text(cx),
                            header.value.read(cx).text(cx),
                        )
                    })
                    .collect(),
                models: form
                    .models
                    .iter()
                    .map(|model| ModelValues {
                        name: model.name.read(cx).text(cx),
                        display_name: model.display_name.read(cx).text(cx),
                        max_completion_tokens: model.max_completion_tokens.read(cx).text(cx),
                        max_output_tokens: model.max_output_tokens.read(cx).text(cx),
                        max_tokens: model.max_tokens.read(cx).text(cx),
                        reasoning_effort: model.reasoning_effort,
                        tool_override: model.tool_override.read(cx).text(cx),
                        default_temperature: model.default_temperature.read(cx).text(cx),
                        extra_beta_headers: model.extra_beta_headers.read(cx).text(cx),
                        mode: model.mode,
                        thinking_budget_tokens: model.thinking_budget_tokens.read(cx).text(cx),
                        supports_tools: model.supports_tools.selected(),
                        supports_images: model.supports_images.selected(),
                        supports_parallel_tool_calls: model.supports_parallel_tool_calls.selected(),
                        supports_prompt_cache_key: model.supports_prompt_cache_key.selected(),
                        supports_chat_completions: model.supports_chat_completions.selected(),
                        supports_thinking: model.supports_thinking.selected(),
                        interleaved_reasoning: model.interleaved_reasoning.selected(),
                        max_tokens_parameter: model.max_tokens_parameter.selected(),
                        supports_prompt_caching: model.supports_prompt_caching.selected(),
                    })
                    .collect(),
            }
        };

        let (provider_name, api_url, api_key, custom_headers, models) =
            match validate_llm_provider_form(&values, cx) {
                Ok(value) => value,
                Err(error) => {
                    if let Mode::AddProvider(form) = &mut self.mode {
                        form.error = Some(error);
                    }
                    cx.notify();
                    return;
                }
            };

        let fs = <dyn fs::Fs>::global(cx);
        cx.spawn_in(window, async move |this, cx| {
            let result = async {
                let provider_id = LanguageModelProviderId(provider_name.clone().into());
                let settings_update = cx.update(|_window, cx| {
                    settings::update_settings_file_with_completion(fs, cx, move |settings, _cx| {
                        let language_models = settings.language_models.get_or_insert_default();
                        match models {
                            ParsedModels::OpenAi(available_models) => {
                                language_models
                                    .openai_compatible
                                    .get_or_insert_default()
                                    .insert(
                                        Arc::from(provider_name.as_str()),
                                        OpenAiCompatibleSettingsContent {
                                            api_url: api_url.clone(),
                                            available_models,
                                            custom_headers: custom_headers.clone(),
                                        },
                                    );
                            }
                            ParsedModels::Anthropic(available_models) => {
                                language_models
                                    .anthropic_compatible
                                    .get_or_insert_default()
                                    .insert(
                                        Arc::from(provider_name.as_str()),
                                        AnthropicCompatibleSettingsContent {
                                            api_url: api_url.clone(),
                                            available_models,
                                            custom_headers: custom_headers.clone(),
                                        },
                                    );
                            }
                        }
                    })
                })?;

                settings_update
                    .await
                    .map_err(|_| anyhow::anyhow!("Settings update was canceled"))??;

                let set_api_key = cx.update(|_window, cx| {
                    let provider = LanguageModelRegistry::read_global(cx)
                        .provider(&provider_id)
                        .ok_or_else(|| anyhow::anyhow!("Provider was not registered"))?;
                    anyhow::Ok(provider.set_api_key(Some(api_key), cx))
                })??;
                set_api_key.await?;

                anyhow::Ok(provider_id)
            }
            .await;

            match result {
                Ok(provider_id) => {
                    cx.update(|window, cx| {
                        this.update(cx, |this, cx| {
                            // The new provider's configuration view is stale as soon as
                            // its settings change, so don't keep a cached one.
                            this.provider_views.remove(&provider_id);
                            this.set_mode(Mode::List, window, cx);
                        })
                    })??;
                }
                Err(error) => {
                    this.update(cx, |this, cx| {
                        if let Mode::AddProvider(form) = &mut this.mode {
                            form.error = Some(error.to_string().into());
                        }
                        cx.notify();
                    })?;
                }
            }

            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    // -- Chrome -------------------------------------------------------------

    fn render_header(&self, mode: &RenderMode, cx: &App) -> ModalHeader {
        match mode {
            RenderMode::List => ModalHeader::new().headline("Configure Models"),
            RenderMode::ProviderConfig(provider_id) => {
                let provider_name = LanguageModelRegistry::read_global(cx)
                    .provider(provider_id)
                    .map(|provider| provider.name().0)
                    .unwrap_or_else(|| provider_id.0.clone());
                ModalHeader::new().headline(provider_name)
            }
            RenderMode::AddProvider(kind) => {
                ModalHeader::new().headline(format!("Add {}-Compatible Provider", kind.label()))
            }
        }
    }

    fn render_footer(&self, mode: &RenderMode, cx: &mut Context<Self>) -> ModalFooter {
        let cancel_binding = KeyBinding::for_action_in(&menu::Cancel, &self.focus_handle, cx)
            .map(|key_binding| key_binding.size(rems_from_px(12_f32)));

        match mode {
            RenderMode::List => ModalFooter::new()
                .start_slot(self.render_add_provider_button(cx))
                .end_slot(
                    h_flex().gap_2().child(
                        Button::new("close-configure-models", "Done")
                            .key_binding(cancel_binding)
                            .on_click(cx.listener(|this, _, _, cx| this.dismiss(cx))),
                    ),
                ),
            RenderMode::ProviderConfig(_) => ModalFooter::new().end_slot(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("cancel-provider-config", "Cancel")
                            .key_binding(cancel_binding)
                            .on_click(cx.listener(|this, _, window, cx| this.go_back(window, cx))),
                    )
                    .child(
                        Button::new("close-configure-models", "Done")
                            .on_click(cx.listener(|this, _, _, cx| this.dismiss(cx))),
                    ),
            ),
            RenderMode::AddProvider(_) => ModalFooter::new().end_slot(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("cancel-add-provider", "Cancel")
                            .key_binding(cancel_binding)
                            .on_click(cx.listener(|this, _, window, cx| this.go_back(window, cx))),
                    )
                    .child(
                        Button::new("save-add-provider", "Save Provider")
                            .style(ButtonStyle::Filled)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.save_add_provider_form(window, cx);
                            })),
                    ),
            ),
        }
    }
}

impl ModalView for ConfigureModelsDialog {
    fn on_before_dismiss(
        &mut self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> DismissDecision {
        // Zed Vela: the dialog is dismissed explicitly, so clicking outside of it
        // doesn't close it.
        DismissDecision::Dismiss(self.dismissed)
    }

    fn fade_out_background(&self) -> bool {
        true
    }
}

impl Focusable for ConfigureModelsDialog {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<DismissEvent> for ConfigureModelsDialog {}

impl Render for ConfigureModelsDialog {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mode = self.render_mode();
        let header = self.render_header(&mode, cx);
        let footer = self.render_footer(&mode, cx);

        let body = match &mode {
            RenderMode::List => self.render_list(window, cx),
            RenderMode::ProviderConfig(provider_id) => {
                let provider_id = provider_id.clone();
                self.render_provider_config(&provider_id, window, cx)
            }
            RenderMode::AddProvider(_) => self.render_add_provider_form(window, cx),
        };

        div()
            .elevation_3(cx)
            .w(rems(52.))
            .key_context("ConfigureModelsDialog")
            .capture_any_mouse_down(cx.listener(|this, _, window, cx| {
                this.focus_handle.focus(window, cx);
            }))
            .on_action(cx.listener(|this, _: &menu::Cancel, window, cx| {
                if matches!(this.mode, Mode::List) {
                    this.dismiss(cx);
                } else {
                    this.go_back(window, cx);
                }
            }))
            .child(
                Modal::new("configure-models", None)
                    .header(header)
                    // Added directly instead of through `Section`, whose built-in
                    // padding is not uniform on all four sides.
                    .child(
                        div()
                            .id("configure-models-scroll")
                            .w_full()
                            .min_w_0()
                            .max_h(vh(0.7, window))
                            .overflow_y_scroll()
                            .track_scroll(&self.scroll_handle)
                            .p(DynamicSpacing::Base12.px(cx))
                            .child(body),
                    )
                    .footer(footer),
            )
    }
}

// -- Form widgets -----------------------------------------------------------

fn render_form_field(
    title: &'static str,
    description: &'static str,
    editor: &Entity<Editor>,
    required: bool,
    cx: &App,
) -> AnyElement {
    v_flex()
        .w_full()
        .gap_1p5()
        .child(
            v_flex()
                .gap_0p5()
                .child(
                    h_flex()
                        .gap_0p5()
                        .child(Label::new(title))
                        .when(required, |this| {
                            this.child(
                                Label::new("*")
                                    .size(LabelSize::Small)
                                    .color(Color::Error)
                                    .mb_2(),
                            )
                        }),
                )
                .child(
                    Label::new(description)
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                ),
        )
        .child(render_compact_input(editor, cx))
        .into_any_element()
}

fn render_compact_input(editor: &Entity<Editor>, cx: &App) -> impl IntoElement {
    let colors = cx.theme().colors();
    let focus_handle = editor.focus_handle(cx).tab_index(0).tab_stop(true);
    h_flex()
        .w_full()
        .min_w_0()
        .h_8()
        .px_2()
        .rounded_md()
        .border_2()
        .border_color(colors.border)
        .bg(colors.editor_background)
        .track_focus(&focus_handle)
        .focus(|style| style.border_color(colors.border_focused))
        .child(editor.clone())
}

fn render_custom_headers(
    form: &LlmProviderForm,
    cx: &mut Context<ConfigureModelsDialog>,
) -> impl IntoElement {
    v_flex()
        .gap_2()
        .child(
            h_flex()
                .justify_between()
                .child(
                    v_flex()
                        .gap_0p5()
                        .child(Label::new("Custom Headers"))
                        .child(
                            Label::new(
                                "Extra HTTP headers sent with every request to this provider.",
                            )
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                        ),
                )
                .child(
                    Button::new("add-custom-header", "Add Header")
                        .start_icon(
                            Icon::new(IconName::Plus)
                                .size(IconSize::XSmall)
                                .color(Color::Muted),
                        )
                        .label_size(LabelSize::Small)
                        .on_click(cx.listener(|this, _, window, cx| {
                            if let Mode::AddProvider(form) = &mut this.mode {
                                form.custom_headers.push(HeaderInput {
                                    name: new_input("Header-Name", None, false, window, cx),
                                    value: new_input("value", None, false, window, cx),
                                });
                            }
                            cx.notify();
                        })),
                ),
        )
        .children(
            form.custom_headers
                .iter()
                .enumerate()
                .map(|(index, header)| {
                    h_flex()
                        .w_full()
                        .min_w_0()
                        .gap_2()
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .child(render_compact_input(&header.name, cx)),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .child(render_compact_input(&header.value, cx)),
                        )
                        .child(
                            IconButton::new(("remove-custom-header", index), IconName::Trash)
                                .icon_size(IconSize::Small)
                                .tooltip(Tooltip::text("Remove Header"))
                                .on_click(cx.listener(move |this, _, _window, cx| {
                                    if let Mode::AddProvider(form) = &mut this.mode
                                        && index < form.custom_headers.len()
                                    {
                                        form.custom_headers.remove(index);
                                    }
                                    cx.notify();
                                })),
                        )
                }),
        )
}

fn render_models_section(
    form: &LlmProviderForm,
    window: &mut Window,
    cx: &mut Context<ConfigureModelsDialog>,
) -> impl IntoElement {
    v_flex()
        .mt_1()
        .gap_2()
        .child(
            h_flex()
                .justify_between()
                .child(Label::new("Models"))
                .child(
                    Button::new("add-model", "Add Model")
                        .start_icon(
                            Icon::new(IconName::Plus)
                                .size(IconSize::XSmall)
                                .color(Color::Muted),
                        )
                        .label_size(LabelSize::Small)
                        .on_click(cx.listener(|this, _, window, cx| {
                            if let Mode::AddProvider(form) = &mut this.mode {
                                form.models.push(ModelInput::new(window, cx));
                            }
                            cx.notify();
                        })),
                ),
        )
        .children(form.models.iter().enumerate().map(|(index, model)| {
            render_model(form.kind, model, index, form.models.len(), window, cx)
        }))
}

fn render_model(
    kind: CompatibleProviderKind,
    model: &ModelInput,
    index: usize,
    model_count: usize,
    window: &mut Window,
    cx: &mut Context<ConfigureModelsDialog>,
) -> AnyElement {
    let is_open_ai = matches!(kind, CompatibleProviderKind::OpenAi);

    v_flex()
        .p_2()
        .gap_2()
        .rounded_sm()
        .border_2()
        .border_dashed()
        .border_color(cx.theme().colors().border.opacity(0.6))
        .bg(cx.theme().colors().element_active.opacity(0.15))
        .child(render_form_field(
            "Model Name",
            "The model's name in the provider's API.",
            &model.name,
            true,
            cx,
        ))
        .child(render_form_field(
            "Display Name",
            "Shown in Zed's model selector. Defaults to the model name.",
            &model.display_name,
            false,
            cx,
        ))
        .when(is_open_ai, |this| {
            this.child(render_form_field(
                "Max Completion Tokens",
                "Maximum completion tokens for OpenAI-compatible requests.",
                &model.max_completion_tokens,
                true,
                cx,
            ))
        })
        .child(render_form_field(
            "Max Output Tokens",
            "The maximum number of tokens the model can output.",
            &model.max_output_tokens,
            true,
            cx,
        ))
        .child(render_form_field(
            "Max Tokens",
            "The model context window size.",
            &model.max_tokens,
            true,
            cx,
        ))
        .when(!is_open_ai, |this| {
            this.child(render_model_mode_selector(model.mode, index, window, cx))
                .when(model.mode == ModelModeInput::Thinking, |this| {
                    this.child(render_form_field(
                        "Thinking Budget Tokens",
                        "Token budget for extended thinking. Leave empty for the provider default.",
                        &model.thinking_budget_tokens,
                        false,
                        cx,
                    ))
                })
                .child(render_form_field(
                    "Tool Override",
                    "A model name to substitute when calling tools, for models that cannot call them.",
                    &model.tool_override,
                    false,
                    cx,
                ))
                .child(render_form_field(
                    "Default Temperature",
                    "Sampling temperature for this model. Leave empty for the provider default.",
                    &model.default_temperature,
                    false,
                    cx,
                ))
                .child(render_form_field(
                    "Extra Beta Headers",
                    "Comma-separated beta header values to send with every request.",
                    &model.extra_beta_headers,
                    false,
                    cx,
                ))
        })
        .child(render_model_capabilities(kind, model, index, window, cx))
        .when(model_count > 1, |this| {
            this.child(
                Button::new(("remove-model", index), "Remove Model")
                    .start_icon(
                        Icon::new(IconName::Trash)
                            .size(IconSize::XSmall)
                            .color(Color::Muted),
                    )
                    .label_size(LabelSize::Small)
                    .style(ButtonStyle::Outlined)
                    .full_width()
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        if let Mode::AddProvider(form) = &mut this.mode
                            && index < form.models.len()
                        {
                            form.models.remove(index);
                        }
                        cx.notify();
                    })),
            )
        })
        .into_any_element()
}

fn render_model_capabilities(
    kind: CompatibleProviderKind,
    model: &ModelInput,
    index: usize,
    window: &mut Window,
    cx: &mut Context<ConfigureModelsDialog>,
) -> impl IntoElement {
    v_flex()
        .gap_1()
        .child(render_capability_checkbox(
            "supports-tools",
            index,
            "Supports tools",
            model.supports_tools,
            |model, state| model.supports_tools = state,
            cx,
        ))
        .child(render_capability_checkbox(
            "supports-images",
            index,
            "Supports images",
            model.supports_images,
            |model, state| model.supports_images = state,
            cx,
        ))
        .when(matches!(kind, CompatibleProviderKind::OpenAi), |this| {
            this.child(render_capability_checkbox(
                "supports-parallel-tool-calls",
                index,
                "Supports parallel_tool_calls",
                model.supports_parallel_tool_calls,
                |model, state| model.supports_parallel_tool_calls = state,
                cx,
            ))
            .child(render_capability_checkbox(
                "supports-prompt-cache-key",
                index,
                "Supports prompt_cache_key",
                model.supports_prompt_cache_key,
                |model, state| model.supports_prompt_cache_key = state,
                cx,
            ))
            .child(render_capability_checkbox(
                "supports-chat-completions",
                index,
                "Supports /chat/completions",
                model.supports_chat_completions,
                |model, state| model.supports_chat_completions = state,
                cx,
            ))
            .when(model.supports_chat_completions.selected(), |this| {
                this.child(render_capability_checkbox(
                    "max-tokens-parameter",
                    index,
                    "Uses max_tokens for output limit",
                    model.max_tokens_parameter,
                    |model, state| model.max_tokens_parameter = state,
                    cx,
                ))
            })
            .child(render_capability_checkbox(
                "supports-thinking",
                index,
                "Supports thinking",
                model.supports_thinking,
                |model, state| model.supports_thinking = state,
                cx,
            ))
            .when(model.supports_thinking.selected(), |this| {
                this.child(render_reasoning_effort_selector(
                    model.reasoning_effort,
                    index,
                    window,
                    cx,
                ))
                .when(model.supports_chat_completions.selected(), |this| {
                    this.child(render_capability_checkbox(
                        "interleaved-reasoning",
                        index,
                        "Preserves thinking in chat history",
                        model.interleaved_reasoning,
                        |model, state| model.interleaved_reasoning = state,
                        cx,
                    ))
                })
            })
        })
        .when(!matches!(kind, CompatibleProviderKind::OpenAi), |this| {
            this.child(render_capability_checkbox(
                "supports-prompt-caching",
                index,
                "Sends prompt caching breakpoints",
                model.supports_prompt_caching,
                |model, state| model.supports_prompt_caching = state,
                cx,
            ))
        })
}

fn render_capability_checkbox(
    id: &'static str,
    index: usize,
    label: &'static str,
    state: ToggleState,
    update: fn(&mut ModelInput, ToggleState),
    cx: &mut Context<ConfigureModelsDialog>,
) -> impl IntoElement {
    Checkbox::new((id, index), state)
        .label(label)
        .on_click(cx.listener(move |this, checked, _window, cx| {
            if let Mode::AddProvider(form) = &mut this.mode
                && let Some(model) = form.models.get_mut(index)
            {
                update(model, *checked);
            }
            cx.notify();
        }))
}

fn render_reasoning_effort_selector(
    selected: OpenAiReasoningEffort,
    index: usize,
    window: &mut Window,
    cx: &mut Context<ConfigureModelsDialog>,
) -> impl IntoElement {
    let dialog = cx.weak_entity();
    let menu = ContextMenu::build(window, cx, move |mut menu, _window, _cx| {
        for effort in OpenAiReasoningEffort::OPENAI_COMPATIBLE_SELECTABLE {
            let is_selected = effort == selected;
            let dialog = dialog.clone();
            menu.push_item(
                ui::ContextMenuEntry::new(effort.label())
                    .toggleable(IconPosition::End, is_selected)
                    .handler(move |_window, cx| {
                        dialog
                            .update(cx, |this, cx| {
                                if let Mode::AddProvider(form) = &mut this.mode
                                    && let Some(model) = form.models.get_mut(index)
                                {
                                    model.reasoning_effort = effort;
                                }
                                cx.notify();
                            })
                            .ok();
                    }),
            );
        }
        menu
    });

    v_flex()
        .gap_1()
        .child(Label::new("Default reasoning effort").size(LabelSize::Small))
        .child(
            DropdownMenu::new(
                ElementId::Name(format!("reasoning-effort-selector-{index}").into()),
                selected.label(),
                menu,
            )
            .style(DropdownStyle::Outlined)
            .trigger_size(ButtonSize::Compact)
            .full_width(true)
            .aria_label("Default reasoning effort"),
        )
}

fn render_model_mode_selector(
    selected: ModelModeInput,
    index: usize,
    window: &mut Window,
    cx: &mut Context<ConfigureModelsDialog>,
) -> impl IntoElement {
    let dialog = cx.weak_entity();
    let menu = ContextMenu::build(window, cx, move |mut menu, _window, _cx| {
        for mode in ModelModeInput::SELECTABLE {
            let is_selected = mode == selected;
            let dialog = dialog.clone();
            menu.push_item(
                ui::ContextMenuEntry::new(mode.label())
                    .toggleable(IconPosition::End, is_selected)
                    .handler(move |_window, cx| {
                        dialog
                            .update(cx, |this, cx| {
                                if let Mode::AddProvider(form) = &mut this.mode
                                    && let Some(model) = form.models.get_mut(index)
                                {
                                    model.mode = mode;
                                }
                                cx.notify();
                            })
                            .ok();
                    }),
            );
        }
        menu
    });

    v_flex()
        .gap_1()
        .child(Label::new("Mode").size(LabelSize::Small))
        .child(
            Label::new("How extended thinking is enabled for this model.")
                .size(LabelSize::Small)
                .color(Color::Muted),
        )
        .child(
            DropdownMenu::new(
                ElementId::Name(format!("model-mode-selector-{index}").into()),
                selected.label(),
                menu,
            )
            .style(DropdownStyle::Outlined)
            .trigger_size(ButtonSize::Compact)
            .full_width(true)
            .aria_label("Mode"),
        )
}

fn render_form_error(error: SharedString) -> impl IntoElement {
    h_flex()
        .w_full()
        .gap_2()
        .child(
            Icon::new(IconName::XCircle)
                .size(IconSize::Small)
                .color(Color::Error),
        )
        .child(Label::new(error).size(LabelSize::Small).color(Color::Error))
}

struct LlmProviderFormValues {
    kind: CompatibleProviderKind,
    provider_name: String,
    api_url: String,
    api_key: String,
    custom_headers: Vec<(String, String)>,
    models: Vec<ModelValues>,
}

struct ModelValues {
    name: String,
    display_name: String,
    max_completion_tokens: String,
    max_output_tokens: String,
    max_tokens: String,
    reasoning_effort: OpenAiReasoningEffort,
    tool_override: String,
    default_temperature: String,
    extra_beta_headers: String,
    mode: ModelModeInput,
    thinking_budget_tokens: String,
    supports_tools: bool,
    supports_images: bool,
    supports_parallel_tool_calls: bool,
    supports_prompt_cache_key: bool,
    supports_chat_completions: bool,
    supports_thinking: bool,
    interleaved_reasoning: bool,
    max_tokens_parameter: bool,
    supports_prompt_caching: bool,
}

enum ParsedModels {
    OpenAi(Vec<OpenAiCompatibleAvailableModel>),
    Anthropic(Vec<AnthropicCompatibleAvailableModel>),
}

fn validate_llm_provider_form(
    values: &LlmProviderFormValues,
    cx: &App,
) -> Result<
    (
        String,
        String,
        String,
        Option<HashMap<String, String>>,
        ParsedModels,
    ),
    SharedString,
> {
    let provider_name = values.provider_name.clone();
    if provider_name.is_empty() {
        return Err("Provider Name cannot be empty".into());
    }

    if LanguageModelRegistry::read_global(cx)
        .providers()
        .iter()
        .any(|provider| {
            provider.id().0.as_ref() == provider_name.as_str()
                || provider.name().0.as_ref() == provider_name.as_str()
        })
    {
        return Err("Provider Name is already taken by another provider".into());
    }

    let api_url = values.api_url.clone();
    if api_url.is_empty() {
        return Err("API URL cannot be empty".into());
    }

    let api_key = values.api_key.clone();
    if api_key.is_empty() {
        return Err("API Key cannot be empty".into());
    }

    let custom_headers = values
        .custom_headers
        .iter()
        .filter(|(name, _)| !name.trim().is_empty())
        .map(|(name, value)| (name.trim().to_string(), value.trim().to_string()))
        .collect::<HashMap<_, _>>();
    let custom_headers = (!custom_headers.is_empty()).then_some(custom_headers);

    let models = match values.kind {
        CompatibleProviderKind::OpenAi => ParsedModels::OpenAi(
            values
                .models
                .iter()
                .map(parse_open_ai_model)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        CompatibleProviderKind::Anthropic => ParsedModels::Anthropic(
            values
                .models
                .iter()
                .map(parse_anthropic_model)
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };

    let mut model_names = HashSet::new();
    let model_names_are_unique = match &models {
        ParsedModels::OpenAi(models) => models
            .iter()
            .all(|model| model_names.insert(model.name.clone())),
        ParsedModels::Anthropic(models) => models
            .iter()
            .all(|model| model_names.insert(model.name.clone())),
    };
    if !model_names_are_unique {
        return Err("Model Names must be unique".into());
    }

    Ok((provider_name, api_url, api_key, custom_headers, models))
}

fn parse_model_name(model: &ModelValues) -> Result<String, SharedString> {
    if model.name.is_empty() {
        return Err("Model Name cannot be empty".into());
    }
    Ok(model.name.clone())
}

fn parse_open_ai_model(
    model: &ModelValues,
) -> Result<OpenAiCompatibleAvailableModel, SharedString> {
    Ok(OpenAiCompatibleAvailableModel {
        name: parse_model_name(model)?,
        display_name: parse_optional_text(&model.display_name),
        max_completion_tokens: Some(parse_u64_field(
            &model.max_completion_tokens,
            "Max Completion Tokens",
        )?),
        max_output_tokens: Some(parse_u64_field(
            &model.max_output_tokens,
            "Max Output Tokens",
        )?),
        max_tokens: parse_u64_field(&model.max_tokens, "Max Tokens")?,
        reasoning_effort: model.supports_thinking.then_some(model.reasoning_effort),
        capabilities: OpenAiCompatibleModelCapabilities {
            tools: model.supports_tools,
            images: model.supports_images,
            parallel_tool_calls: model.supports_parallel_tool_calls,
            prompt_cache_key: model.supports_prompt_cache_key,
            chat_completions: model.supports_chat_completions,
            interleaved_reasoning: model.supports_thinking
                && model.supports_chat_completions
                && model.interleaved_reasoning,
            max_tokens_parameter: model.supports_chat_completions && model.max_tokens_parameter,
        },
    })
}

fn parse_anthropic_model(
    model: &ModelValues,
) -> Result<AnthropicCompatibleAvailableModel, SharedString> {
    Ok(AnthropicCompatibleAvailableModel {
        name: parse_model_name(model)?,
        display_name: parse_optional_text(&model.display_name),
        max_tokens: parse_u64_field(&model.max_tokens, "Max Tokens")?,
        tool_override: parse_optional_text(&model.tool_override),
        max_output_tokens: Some(parse_u64_field(
            &model.max_output_tokens,
            "Max Output Tokens",
        )?),
        default_temperature: parse_optional_f32(&model.default_temperature, "Default Temperature")?,
        extra_beta_headers: parse_list(&model.extra_beta_headers),
        mode: match model.mode {
            ModelModeInput::Default => None,
            ModelModeInput::Thinking => Some(ModelMode::Thinking {
                budget_tokens: parse_optional_u32(
                    &model.thinking_budget_tokens,
                    "Thinking Budget Tokens",
                )?,
            }),
            ModelModeInput::Adaptive => Some(ModelMode::Adaptive),
        },
        capabilities: AnthropicCompatibleModelCapabilities {
            tools: model.supports_tools,
            images: model.supports_images,
            prompt_caching: model.supports_prompt_caching,
        },
    })
}

fn parse_optional_text(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn parse_optional_f32(value: &str, name: &str) -> Result<Option<f32>, SharedString> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    value
        .parse::<f32>()
        .map(Some)
        .map_err(|_| format!("{name} must be a number").into())
}

fn parse_optional_u32(value: &str, name: &str) -> Result<Option<u32>, SharedString> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    value
        .parse::<u32>()
        .map(Some)
        .map_err(|_| format!("{name} must be a whole number").into())
}

fn parse_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_u64_field(value: &str, name: &str) -> Result<u64, SharedString> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("{name} must be a number").into())
}
