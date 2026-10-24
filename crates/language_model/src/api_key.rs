use anyhow::{Result, anyhow};
use credentials_provider::CredentialsProvider;
use env_var::EnvVar;
use futures::{FutureExt, channel::oneshot, future};
use gpui::{App, AsyncApp, Context, Global, SharedString, Subscription, Task};
use gpui_util::ResultExt as _;
use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::AuthenticateError;

/// Manages a single API key for a language model provider. API keys either come from environment
/// variables or the system keychain.
///
/// Keys from the system keychain are associated with a provider URL, and optionally with a
/// provider scope, so multiple provider entries can share a URL while keeping distinct keys.
pub struct ApiKeyState {
    pub url: SharedString,
    /// An additional identifier (typically the provider name) included in the keychain key.
    credentials_scope: Option<SharedString>,
    env_var: EnvVar,
    load_status: LoadStatus,
    load_task: Option<ApiKeyLoad>,
}

#[derive(Debug, Clone)]
pub enum LoadStatus {
    NotLoaded,
    NotPresent,
    Error(String),
    Loaded(ApiKey),
}

#[derive(Debug, Clone)]
pub struct ApiKey {
    source: ApiKeySource,
    key: Arc<str>,
}

pub fn set_credentials_read_gate<G: Global>(is_disabled: fn(&App) -> bool, cx: &mut App) {
    cx.set_global(CredentialsReadGate {
        is_disabled,
        subscribe: |cx, is_disabled, sender| {
            let mut sender = Some(sender);
            cx.observe_global::<G>(move |cx| {
                if !is_disabled(cx)
                    && let Some(sender) = sender.take()
                {
                    sender.send(()).ok();
                }
            })
        },
    });
}

pub async fn read_credentials(
    provider: &dyn CredentialsProvider,
    url: &str,
    cx: &AsyncApp,
) -> Result<Option<(String, Vec<u8>)>> {
    while let Some((receiver, _subscription)) = cx.update(|cx| {
        let gate = *cx.try_global::<CredentialsReadGate>()?;
        if !(gate.is_disabled)(cx) {
            return None;
        }
        let (sender, receiver) = oneshot::channel();
        let subscription = (gate.subscribe)(cx, gate.is_disabled, sender);
        Some((receiver, subscription))
    }) {
        receiver.await?;
    }
    provider.read_credentials(url, cx).await
}

impl ApiKeyState {
    pub fn new(url: SharedString, env_var: EnvVar) -> Self {
        Self::new_with_scope(url, None, env_var)
    }

    /// Creates a key state whose keychain entry is scoped to `scope` in addition to the URL.
    ///
    /// This allows multiple providers to use the same API URL with different API keys.
    pub fn new_with_scope(url: SharedString, scope: Option<SharedString>, env_var: EnvVar) -> Self {
        Self {
            url,
            credentials_scope: scope,
            env_var,
            load_status: LoadStatus::NotLoaded,
            load_task: None,
        }
    }

    /// The key used to address this provider's API key in the system keychain.
    fn credentials_key(&self, url: &str) -> SharedString {
        match &self.credentials_scope {
            // Length-prefix the scope so a scope containing `:` cannot collide with a
            // different scope/URL split.
            Some(scope) => SharedString::new(format!(
                "zed-llm-provider:{}:{scope}:{url}",
                scope.as_str().len()
            )),
            None => SharedString::new(url),
        }
    }

    pub fn has_key(&self) -> bool {
        matches!(self.load_status, LoadStatus::Loaded { .. })
    }

    pub fn env_var_name(&self) -> &SharedString {
        &self.env_var.name
    }

    pub fn is_from_env_var(&self) -> bool {
        match &self.load_status {
            LoadStatus::Loaded(ApiKey {
                source: ApiKeySource::EnvVar { .. },
                ..
            }) => true,
            _ => false,
        }
    }

    /// Get the stored API key, verifying that it is associated with the URL. Returns `None` if
    /// there is no key or for URL mismatches, and the mismatch case is logged.
    ///
    /// To avoid URL mismatches, expects that `load_if_needed` or `handle_url_change` has been
    /// called with this URL.
    pub fn key(&self, url: &str) -> Option<Arc<str>> {
        let api_key = match &self.load_status {
            LoadStatus::Loaded(api_key) => api_key,
            _ => return None,
        };
        if url == self.url.as_str() {
            Some(api_key.key.clone())
        } else if let ApiKeySource::EnvVar(var_name) = &api_key.source {
            log::warn!(
                "{} is now being used with URL {}, when initially it was used with URL {}",
                var_name,
                url,
                self.url
            );
            Some(api_key.key.clone())
        } else {
            // bug case because load_if_needed should be called whenever the url may have changed
            log::error!(
                "bug: Attempted to use API key associated with URL {} instead with URL {}",
                self.url,
                url
            );
            None
        }
    }

    /// Set or delete the API key in the system keychain.
    pub fn store<Ent: 'static>(
        &mut self,
        url: SharedString,
        key: Option<String>,
        get_this: impl Fn(&mut Ent) -> &mut Self + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) -> Task<Result<()>> {
        if self.is_from_env_var() {
            return Task::ready(Err(anyhow!(
                "bug: attempted to store API key in system keychain when API key is from env var",
            )));
        }
        if self.url != url {
            self.url = url.clone();
            self.load_status = LoadStatus::NotLoaded;
            self.load_task = None;
            cx.notify();
        }

        let credentials_key = self.credentials_key(&url);
        // Legacy keys live under the bare URL. Remove that entry as well when a key is
        // deleted, or it would be migrated back in on the next load.
        let legacy_url = (self.credentials_scope.is_some() && key.is_none()).then(|| url.clone());
        cx.spawn(async move |ent, cx| {
            if let Some(key) = &key {
                provider
                    .write_credentials(&credentials_key, "Bearer", key.as_bytes(), cx)
                    .await
                    .log_err();
            } else {
                provider
                    .delete_credentials(&credentials_key, cx)
                    .await
                    .log_err();
                if let Some(legacy_url) = &legacy_url {
                    provider.delete_credentials(legacy_url, cx).await.log_err();
                }
            }
            ent.update(cx, |ent, cx| {
                let key_state = get_this(ent);
                if key_state.url != url {
                    return;
                }
                key_state.load_task = None;
                key_state.load_status = match &key {
                    Some(key) => LoadStatus::Loaded(ApiKey {
                        source: ApiKeySource::SystemKeychain,
                        key: key.as_str().into(),
                    }),
                    None => LoadStatus::NotPresent,
                };
                cx.notify();
            })
        })
    }

    /// Reloads the API key if the current API key is associated with a different URL.
    ///
    /// Note that it is not efficient to use this or `load_if_needed` with multiple URLs
    /// interchangeably - URL change should correspond to some user initiated change.
    pub fn handle_url_change<Ent: 'static>(
        &mut self,
        url: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + Clone + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) {
        if url != self.url {
            if !self.is_from_env_var() {
                // loading will continue even though this result task is dropped
                let _task = self.load_if_needed(url, get_this, provider, cx);
            }
        }
    }

    pub fn load_cached<Ent: 'static>(
        &mut self,
        url: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + Clone + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) -> Task<Result<(), AuthenticateError>> {
        if let LoadStatus::NotPresent | LoadStatus::Error(_) = &self.load_status
            && self.url == url
        {
            return Task::ready(Ok(()));
        }

        self.load_if_needed(url, get_this, provider, cx)
    }

    /// If needed, loads the API key associated with the given URL from the system keychain. When a
    /// non-empty environment variable is provided, it will be used instead. If called when an API
    /// key was already loaded for a different URL, that key will be cleared before loading.
    ///
    /// Dropping the returned Task does not cancel key loading.
    pub fn load_if_needed<Ent: 'static>(
        &mut self,
        url: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + Clone + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) -> Task<Result<(), AuthenticateError>> {
        if let LoadStatus::Loaded { .. } = &self.load_status
            && self.url == url
        {
            return Task::ready(Ok(()));
        }

        if let Some(key) = &self.env_var.value
            && !key.is_empty()
        {
            let api_key = ApiKey::from_env(self.env_var.name.clone(), key);
            self.url = url;
            self.load_status = LoadStatus::Loaded(api_key);
            self.load_task = None;
            cx.notify();
            return Task::ready(Ok(()));
        }

        let completion = if let Some(load_task) = &self.load_task {
            load_task.completion.clone()
        } else {
            let credentials_key = self.credentials_key(&url);
            let load_task =
                Self::load(url.clone(), credentials_key, get_this.clone(), provider, cx);
            let completion = load_task.completion.clone();
            self.url = url;
            self.load_status = LoadStatus::NotPresent;
            self.load_task = Some(load_task);
            cx.notify();
            completion
        };

        cx.spawn(async move |ent, cx| {
            completion.await.ok();
            ent.update(cx, |ent, _cx| {
                get_this(ent).load_status.clone().into_authenticate_result()
            })
            .ok();
            Ok(())
        })
    }

    fn load<Ent: 'static>(
        url: SharedString,
        credentials_key: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + Clone + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &Context<Ent>,
    ) -> ApiKeyLoad {
        let (sender, receiver) = oneshot::channel();
        let task = cx.spawn({
            async move |ent, cx| {
                let mut load_status =
                    ApiKey::load_from_system_keychain_impl(&credentials_key, provider.as_ref(), cx)
                        .await;

                // Older builds stored compatible-provider keys under the bare URL. Move that key
                // to the scoped entry the first time it is loaded.
                if matches!(load_status, LoadStatus::NotPresent)
                    && credentials_key.as_str() != url.as_str()
                    && let LoadStatus::Loaded(api_key) =
                        ApiKey::load_from_system_keychain_impl(&url, provider.as_ref(), cx).await
                {
                    provider
                        .write_credentials(&credentials_key, "Bearer", api_key.key().as_bytes(), cx)
                        .await
                        .log_err();
                    load_status = LoadStatus::Loaded(api_key);
                }

                ent.update(cx, |ent, cx| {
                    let this = get_this(ent);
                    this.url = url;
                    this.load_status = load_status;
                    this.load_task = None;
                    cx.notify();
                })
                .ok();
                sender.send(()).ok();
            }
        });
        ApiKeyLoad {
            _task: task,
            completion: receiver.shared(),
        }
    }
}

impl ApiKey {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn from_env(env_var_name: SharedString, key: &str) -> Self {
        Self {
            source: ApiKeySource::EnvVar(env_var_name),
            key: key.into(),
        }
    }

    pub async fn load_from_system_keychain(
        url: &str,
        credentials_provider: &dyn CredentialsProvider,
        cx: &AsyncApp,
    ) -> Result<Self, AuthenticateError> {
        Self::load_from_system_keychain_impl(url, credentials_provider, cx)
            .await
            .into_authenticate_result()
    }

    async fn load_from_system_keychain_impl(
        url: &str,
        credentials_provider: &dyn CredentialsProvider,
        cx: &AsyncApp,
    ) -> LoadStatus {
        if url.is_empty() {
            return LoadStatus::NotPresent;
        }
        let read_result = read_credentials(credentials_provider, url, cx).await;
        let api_key = match read_result {
            Ok(Some((_, api_key))) => api_key,
            Ok(None) => return LoadStatus::NotPresent,
            Err(err) => return LoadStatus::Error(err.to_string()),
        };
        let key = match str::from_utf8(&api_key) {
            Ok(key) => key,
            Err(_) => return LoadStatus::Error(format!("API key for URL {url} is not utf8")),
        };
        LoadStatus::Loaded(Self {
            source: ApiKeySource::SystemKeychain,
            key: key.into(),
        })
    }
}

impl LoadStatus {
    fn into_authenticate_result(self) -> Result<ApiKey, AuthenticateError> {
        match self {
            LoadStatus::Loaded(api_key) => Ok(api_key),
            LoadStatus::NotLoaded | LoadStatus::NotPresent => {
                Err(AuthenticateError::CredentialsNotFound)
            }
            LoadStatus::Error(err) => Err(AuthenticateError::Other(anyhow!(err))),
        }
    }
}

#[derive(Clone, Copy)]
struct CredentialsReadGate {
    is_disabled: fn(&App) -> bool,
    subscribe: fn(&mut App, fn(&App) -> bool, oneshot::Sender<()>) -> Subscription,
}

impl Global for CredentialsReadGate {}

struct ApiKeyLoad {
    _task: Task<()>,
    completion: future::Shared<oneshot::Receiver<()>>,
}

#[derive(Debug, Clone)]
enum ApiKeySource {
    EnvVar(SharedString),
    SystemKeychain,
}

impl Display for ApiKeySource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiKeySource::EnvVar(var) => write!(f, "environment variable {var}"),
            ApiKeySource::SystemKeychain => write!(f, "system keychain"),
        }
    }
}
