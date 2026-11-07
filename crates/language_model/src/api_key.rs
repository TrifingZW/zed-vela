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

        let completion = if let Some(load_task) = &self.load_task
            && self.url == url
        {
            load_task.completion.clone()
        } else {
            let credentials_key = self.credentials_key(&url);
            let load_task =
                Self::load(url.clone(), credentials_key, get_this.clone(), provider, cx);
            let completion = load_task.completion.clone();
            self.url = url;
            self.load_status = LoadStatus::NotLoaded;
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

#[cfg(test)]
mod tests {
    use super::*;
    use credentials_provider::test_support::FakeCredentialsProvider;
    use gpui::{AppContext as _, Entity, TestAppContext};

    const URL: &str = "provider";
    const OTHER_URL: &str = "other-provider";
    const SCOPE: &str = "First";

    #[gpui::test]
    async fn test_cached_miss_and_explicit_retry(cx: &mut TestAppContext) {
        for fail_reads in [false, true] {
            let test = KeyTest::new(&[], cx);
            test.provider.state.lock().fail_reads = fail_reads;
            test.load(URL, cx).await.expect("initial load");
            {
                let mut credentials = test.provider.state.lock();
                credentials.fail_reads = false;
                credentials
                    .keys
                    .insert(URL.to_string(), b"late-key".to_vec());
            }
            test.load(URL, cx).await.expect("cached miss");
            assert_eq!(test.key(URL, cx), None);
            test.provider.assert_reads(&[URL]);
            let url = SharedString::from(URL);
            let retry = test.state.update(cx, |state, cx| {
                state.load_if_needed(url, |state| state, test.provider.clone(), cx)
            });
            retry.await.expect("explicit retry");
            assert_eq!(test.key(URL, cx).as_deref(), Some("late-key"));
            test.load(URL, cx).await.expect("cached key");
            test.provider.assert_reads(&[URL, URL]);
        }
    }

    #[gpui::test]
    async fn test_disabled_gate_and_superseded_loads(cx: &mut TestAppContext) {
        cx.update(|cx| {
            set_credentials_read_gate::<DisabledAi>(|cx| cx.global::<DisabledAi>().0, cx);
        });
        let test = KeyTest::new(&[(OTHER_URL, "latest-key")], cx);
        set_disabled_ai(true, cx);
        let superseded = test.load(URL, cx);
        cx.run_until_parked();
        drop(test.load(OTHER_URL, cx));
        cx.run_until_parked();
        test.provider.assert_reads(&[]);
        set_disabled_ai(false, cx);
        set_disabled_ai(true, cx);
        cx.run_until_parked();
        test.provider.assert_reads(&[]);
        let release = test.provider.defer_read(OTHER_URL);
        set_disabled_ai(false, cx);
        cx.run_until_parked();
        drop(test.load(OTHER_URL, cx));
        set_disabled_ai(true, cx);
        cx.run_until_parked();
        test.provider.assert_reads(&[OTHER_URL]);
        release.send(()).expect("started read remains live");
        cx.run_until_parked();
        superseded.await.expect("superseded load completes");
        assert_eq!(test.key(OTHER_URL, cx).as_deref(), Some("latest-key"));
        test.provider.assert_reads(&[OTHER_URL]);
    }

    #[gpui::test]
    async fn test_store_cancels_stale_read(cx: &mut TestAppContext) {
        for stored_key in [Some("saved-key"), None] {
            let test = KeyTest::new(&[(URL, "old-key")], cx);
            let release = test.provider.defer_read(URL);
            let load = test.load(URL, cx);
            cx.run_until_parked();
            test.store(stored_key, cx).await.expect("mutation");
            cx.run_until_parked();
            assert_eq!(release.send(()), Err(()));
            load.await.expect("cancelled read completes");
            test.load(URL, cx).await.expect("cached mutation");
            assert_eq!(test.key(URL, cx).as_deref(), stored_key);
            test.provider.assert_reads(&[URL]);
        }
    }

    #[gpui::test]
    async fn test_deferred_mutations_follow_url_identity(cx: &mut TestAppContext) {
        for stored_key in [Some("saved-key"), None] {
            let test = KeyTest::new(&[(URL, "old-key"), (OTHER_URL, "other-key")], cx);
            test.load(URL, cx).await.expect("initial load");
            let release = test.provider.defer_next_mutation();
            let mutation = test.store(stored_key, cx);
            cx.run_until_parked();
            let release_read = test.provider.defer_read(OTHER_URL);
            let load = test.load(OTHER_URL, cx);
            cx.run_until_parked();
            release.send(()).expect("pending mutation");
            mutation.await.expect("mutation for inactive URL");
            drop(test.load(OTHER_URL, cx));
            cx.run_until_parked();
            assert_eq!(test.key(OTHER_URL, cx), None);
            release_read.send(()).expect("other URL read remains live");
            load.await.expect("other URL load");
            assert_eq!(test.key(OTHER_URL, cx).as_deref(), Some("other-key"));
            assert_eq!(
                test.provider.state.lock().keys.get(URL).map(Vec::as_slice),
                stored_key.map(str::as_bytes)
            );
            test.provider.assert_reads(&[URL, OTHER_URL]);
        }
    }

    #[gpui::test]
    async fn scoped_keys_use_length_prefixed_credential_entries(cx: &mut TestAppContext) {
        let provider = Arc::new(FakeCredentialsProvider::default());
        let state = new_state(Some(SCOPE), cx);
        let url = SharedString::from(URL);
        state
            .update(cx, |state, cx| {
                state.store(
                    url.clone(),
                    Some("scoped-key".into()),
                    |state| state,
                    provider.clone(),
                    cx,
                )
            })
            .await
            .expect("store");

        let scoped_key = format!("zed-llm-provider:{}:{SCOPE}:{URL}", SCOPE.len());
        assert_eq!(
            provider
                .state
                .lock()
                .keys
                .get(&scoped_key)
                .map(Vec::as_slice),
            Some(b"scoped-key".as_slice())
        );
        assert!(
            provider.state.lock().keys.get(URL).is_none(),
            "scoped keys must not be written under the bare URL"
        );

        let reloaded = new_state(Some(SCOPE), cx);
        reloaded
            .update(cx, |state, cx| {
                state.load_if_needed(url, |state| state, provider.clone(), cx)
            })
            .await
            .expect("load");
        assert_eq!(
            reloaded.read_with(cx, |state, _| state.key(URL)).as_deref(),
            Some("scoped-key")
        );
    }

    #[gpui::test]
    async fn distinct_scopes_keep_distinct_keys(cx: &mut TestAppContext) {
        let provider = Arc::new(FakeCredentialsProvider::default());
        let first = new_state(Some(SCOPE), cx);
        let second = new_state(Some("Second"), cx);
        let url = SharedString::from(URL);
        for (state, key) in [(&first, "first-key"), (&second, "second-key")] {
            state
                .update(cx, |state, cx| {
                    state.store(
                        url.clone(),
                        Some(key.into()),
                        |state| state,
                        provider.clone(),
                        cx,
                    )
                })
                .await
                .expect("store");
        }

        let reloaded_first = new_state(Some(SCOPE), cx);
        reloaded_first
            .update(cx, |state, cx| {
                state.load_if_needed(url.clone(), |state| state, provider.clone(), cx)
            })
            .await
            .expect("load first");
        assert_eq!(
            reloaded_first
                .read_with(cx, |state, _| state.key(URL))
                .as_deref(),
            Some("first-key")
        );

        let reloaded_second = new_state(Some("Second"), cx);
        reloaded_second
            .update(cx, |state, cx| {
                state.load_if_needed(url, |state| state, provider.clone(), cx)
            })
            .await
            .expect("load second");
        assert_eq!(
            reloaded_second
                .read_with(cx, |state, _| state.key(URL))
                .as_deref(),
            Some("second-key")
        );
    }

    #[gpui::test]
    async fn legacy_url_keys_migrate_to_the_scoped_entry(cx: &mut TestAppContext) {
        let provider = Arc::new(FakeCredentialsProvider::default());
        provider
            .state
            .lock()
            .keys
            .insert(URL.to_string(), b"legacy-key".to_vec());
        let state = new_state(Some(SCOPE), cx);
        let url = SharedString::from(URL);
        state
            .update(cx, |state, cx| {
                state.load_if_needed(url, |state| state, provider.clone(), cx)
            })
            .await
            .expect("load");

        assert_eq!(
            state.read_with(cx, |state, _| state.key(URL)).as_deref(),
            Some("legacy-key")
        );
        let scoped_key = format!("zed-llm-provider:{}:{SCOPE}:{URL}", SCOPE.len());
        assert_eq!(
            provider
                .state
                .lock()
                .keys
                .get(&scoped_key)
                .map(Vec::as_slice),
            Some(b"legacy-key".as_slice())
        );
        provider.assert_reads(&[scoped_key.as_str(), URL]);
    }

    #[gpui::test]
    async fn deleting_a_scoped_key_also_removes_the_legacy_entry(cx: &mut TestAppContext) {
        let provider = Arc::new(FakeCredentialsProvider::default());
        let scoped_key = format!("zed-llm-provider:{}:{SCOPE}:{URL}", SCOPE.len());
        {
            let mut credentials = provider.state.lock();
            credentials
                .keys
                .insert(URL.to_string(), b"legacy-key".to_vec());
            credentials
                .keys
                .insert(scoped_key.clone(), b"legacy-key".to_vec());
        }
        let state = new_state(Some(SCOPE), cx);
        let url = SharedString::from(URL);
        state
            .update(cx, |state, cx| {
                state.store(url, None, |state| state, provider.clone(), cx)
            })
            .await
            .expect("delete");

        let credentials = provider.state.lock();
        assert!(credentials.keys.get(&scoped_key).is_none());
        assert!(
            credentials.keys.get(URL).is_none(),
            "the legacy entry would be migrated back on the next load"
        );
    }

    type LoadTask = Task<Result<(), AuthenticateError>>;

    struct DisabledAi(bool);

    impl Global for DisabledAi {}

    struct KeyTest {
        state: Entity<ApiKeyState>,
        provider: Arc<FakeCredentialsProvider>,
    }

    impl KeyTest {
        fn new(keys: &[(&str, &str)], cx: &mut TestAppContext) -> Self {
            let provider = Arc::new(FakeCredentialsProvider::default());
            provider.state.lock().keys.extend(
                keys.iter()
                    .map(|(url, key)| (url.to_string(), key.as_bytes().to_vec())),
            );
            Self {
                state: new_state(None, cx),
                provider,
            }
        }

        fn load(&self, url: &'static str, cx: &mut TestAppContext) -> LoadTask {
            let url = SharedString::from(url);
            self.state.update(cx, |state, cx| {
                state.load_cached(url, |state| state, self.provider.clone(), cx)
            })
        }

        fn store(&self, key: Option<&str>, cx: &mut TestAppContext) -> Task<Result<()>> {
            let url = SharedString::from(URL);
            let key = key.map(String::from);
            self.state.update(cx, |state, cx| {
                state.store(url, key, |state| state, self.provider.clone(), cx)
            })
        }

        fn key(&self, url: &str, cx: &TestAppContext) -> Option<Arc<str>> {
            self.state.read_with(cx, |state, _| state.key(url))
        }
    }

    fn new_state(scope: Option<&'static str>, cx: &mut TestAppContext) -> Entity<ApiKeyState> {
        cx.new(|_| {
            ApiKeyState::new_with_scope(
                SharedString::default(),
                scope.map(SharedString::new_static),
                EnvVar {
                    name: SharedString::new_static("TEST_API_KEY"),
                    value: None,
                },
            )
        })
    }

    fn set_disabled_ai(disabled: bool, cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_global(DisabledAi(disabled)));
    }
}
