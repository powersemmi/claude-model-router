//! The bounded store of Gemini thought signatures.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

type Inner = (HashMap<Box<str>, Arc<str>>, VecDeque<Box<str>>);

/// Bounded map from a tool_use id to the Gemini thought signature of that call. Claude Code
/// drops fields it does not know, so the router keeps the signatures and puts them back on the
/// next request. The oldest entry goes first when the capacity is reached.
pub struct SignatureStore {
    capacity: usize,
    inner: Mutex<Inner>,
}

impl SignatureStore {
    /// A store holding at most `capacity` signatures (at least one).
    pub fn new(capacity: usize) -> Self {
        SignatureStore {
            capacity: capacity.max(1),
            inner: Mutex::new((HashMap::new(), VecDeque::new())),
        }
    }

    /// Stores `sig` for `id`. An existing entry for `id` is replaced and keeps its age.
    pub fn put(&self, id: &str, sig: &str) {
        let sig: Arc<str> = Arc::from(sig);
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (map, order) = &mut *g;
        if let Some(slot) = map.get_mut(id) {
            *slot = sig;
            return;
        }
        if map.len() >= self.capacity {
            if let Some(old) = order.pop_front() {
                map.remove(&old);
            }
        }
        let key: Box<str> = id.into();
        order.push_back(key.clone());
        map.insert(key, sig);
    }

    /// The signature stored for `id`.
    pub fn get(&self, id: &str) -> Option<Arc<str>> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.0.get(id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get() {
        let s = SignatureStore::new(4);
        assert!(s.get("a").is_none());
        s.put("a", "sig-a");
        assert_eq!(&*s.get("a").unwrap(), "sig-a");
    }

    #[test]
    fn evicts_oldest_first() {
        let s = SignatureStore::new(2);
        s.put("a", "1");
        s.put("b", "2");
        s.put("c", "3");
        assert!(s.get("a").is_none());
        assert_eq!(&*s.get("b").unwrap(), "2");
        s.put("d", "4");
        assert!(s.get("b").is_none());
        assert!(s.get("c").is_some() && s.get("d").is_some());
    }

    #[test]
    fn overwrite_keeps_size() {
        let s = SignatureStore::new(2);
        s.put("a", "1");
        s.put("b", "2");
        s.put("a", "9");
        assert_eq!(&*s.get("a").unwrap(), "9");
        assert!(s.get("b").is_some());
        s.put("c", "3");
        assert!(s.get("a").is_none());
    }
}
