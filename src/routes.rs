//! The routing table: exact model name to provider, with a default for everything else.

use std::collections::HashMap;
use std::sync::Arc;

use crate::provider::Provider;

/// Maps a request's model name to the provider that serves it.
pub struct Routes {
    default: Arc<dyn Provider>,
    by_model: HashMap<String, Arc<dyn Provider>>,
}

impl Routes {
    /// A table whose default provider serves every model that has no entry.
    pub fn new(default: Arc<dyn Provider>) -> Self {
        Self {
            default,
            by_model: HashMap::new(),
        }
    }

    /// Sends requests naming exactly `model` to `provider`.
    pub fn add(&mut self, model: impl Into<String>, provider: Arc<dyn Provider>) {
        self.by_model.insert(model.into(), provider);
    }

    /// The provider for `model`; the default when the model is absent or unlisted.
    pub fn resolve(&self, model: Option<&str>) -> &Arc<dyn Provider> {
        model
            .and_then(|m| self.by_model.get(m))
            .unwrap_or(&self.default)
    }

    /// Every listed model with its provider's name, sorted by model, for the startup log.
    pub fn models(&self) -> impl Iterator<Item = (&str, &'static str)> {
        let mut v: Vec<_> = self
            .by_model
            .iter()
            .map(|(m, p)| (m.as_str(), p.name()))
            .collect();
        v.sort_unstable();
        v.into_iter()
    }
}
