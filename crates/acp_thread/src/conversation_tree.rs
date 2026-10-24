//! A tree of conversation branches layered on top of a thread's linear
//! transcript.
//!
//! [`crate::AcpThread`] keeps the active path materialized in its `entries`
//! vector because every existing consumer (streaming, tool call updates, the
//! list view) indexes into that vector. Branches that are not active keep their
//! entries here instead. Entries are moved between the two stores rather than
//! cloned: [`AgentThreadEntry`] owns GPUI entities, task handles, and terminal
//! handles, so cloning is neither cheap nor always possible.
//!
//! The invariant is that for every node on the active path
//! `root -> .. -> active`, `entry_count` is the number of entries the node
//! contributes to `AcpThread::entries`, and `stored_entries` is empty. For every
//! other node, `stored_entries.len() == entry_count`.
//!
//! Branch switching and persistence are only offered for agents that can
//! archive and restore branches (see
//! [`crate::AgentConnection::supports_conversation_branches`]) — currently the
//! native agent, whose linear session is truncated and refilled when branches
//! switch. ACP agents are linear: their sessions can only be truncated, never
//! refilled, so the UI hides the branch graph and the tree is never persisted
//! for them.

use crate::{AgentThreadEntry, ClientUserMessageId};
use anyhow::{Result, bail};
use collections::IndexMap;
use gpui::SharedString;
use serde::{Deserialize, Serialize};
use std::mem;

/// Identifies a node in a [`ConversationTree`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConversationNodeId(u64);

impl ConversationNodeId {
    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }
}

/// A read-only projection used by the UI to render branch switchers.
#[derive(Clone, Debug)]
pub struct ConversationBranch {
    pub id: ConversationNodeId,
    pub parent: Option<ConversationNodeId>,
    pub label: SharedString,
    pub entry_count: usize,
    pub is_active: bool,
    pub is_on_active_path: bool,
}

#[derive(Debug)]
pub struct ConversationNode {
    id: ConversationNodeId,
    parent: Option<ConversationNodeId>,
    children: Vec<ConversationNodeId>,
    label: SharedString,
    entry_count: usize,
    stored_entries: Vec<AgentThreadEntry>,
    awaiting_user_message: bool,
}

impl ConversationNode {
    pub(crate) fn parent(&self) -> Option<ConversationNodeId> {
        self.parent
    }

    pub(crate) fn is_awaiting_user_message(&self) -> bool {
        self.awaiting_user_message
    }
}

/// Serializable structure of a [`ConversationTree`], without any entries.
/// Persisted alongside the thread so the tree survives restarts; the entries
/// of every branch are reconstructed by replaying the per-node messages that
/// the agent stores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedConversationTree {
    pub active: u64,
    pub nodes: Vec<SerializedConversationNode>,
}

/// A single serialized node. `start_client_id` is only meaningful for nodes on
/// the active path: it is the client message id that starts the node's segment
/// in the linear transcript, used by the agent to split its live messages per
/// node when persisting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedConversationNode {
    pub id: u64,
    pub parent: Option<u64>,
    pub label: String,
    #[serde(default)]
    pub awaiting_user_message: bool,
    #[serde(default)]
    pub start_client_id: Option<ClientUserMessageId>,
}

/// Whether entries beyond a truncation boundary are kept as an inactive branch
/// or discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConversationTruncation {
    /// Keep the truncated suffix as a sibling branch. Used by `rewind`, which
    /// backs editing a previous message: the old turn becomes a branch instead
    /// of disappearing.
    Preserve,
    /// Discard the truncated suffix entirely. Used when a prompt that never
    /// produced a response is rolled back.
    Discard,
}

#[derive(Debug)]
pub struct ConversationTree {
    nodes: IndexMap<ConversationNodeId, ConversationNode>,
    root: ConversationNodeId,
    active: ConversationNodeId,
    next_id: u64,
}

impl Default for ConversationTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationTree {
    pub fn new() -> Self {
        let root = ConversationNodeId(0);
        let mut nodes = IndexMap::default();
        nodes.insert(
            root,
            ConversationNode {
                id: root,
                parent: None,
                children: Vec::new(),
                label: "Main".into(),
                entry_count: 0,
                stored_entries: Vec::new(),
                awaiting_user_message: false,
            },
        );
        Self {
            nodes,
            root,
            active: root,
            next_id: 1,
        }
    }

    pub fn active_node(&self) -> ConversationNodeId {
        self.active
    }

    pub(crate) fn node(&self, id: ConversationNodeId) -> Option<&ConversationNode> {
        self.nodes.get(&id)
    }

    /// All nodes in creation order, which guarantees parents precede children.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (ConversationNodeId, &ConversationNode)> {
        self.nodes.iter().map(|(id, node)| (*id, node))
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of entries the node contributes to the active transcript.
    pub(crate) fn entry_count(&self, id: ConversationNodeId) -> usize {
        self.nodes.get(&id).map_or(0, |node| node.entry_count)
    }

    pub(crate) fn label(&self, id: ConversationNodeId) -> Option<&SharedString> {
        self.nodes.get(&id).map(|node| &node.label)
    }

    pub(crate) fn path_to(&self, id: ConversationNodeId) -> Vec<ConversationNodeId> {
        let mut path = Vec::new();
        let mut current = Some(id);
        while let Some(node_id) = current {
            let Some(node) = self.nodes.get(&node_id) else {
                break;
            };
            path.push(node_id);
            if path.len() > self.nodes.len() {
                // Defensive: parent pointers are only written by this module,
                // but a cycle must never hang rendering.
                break;
            }
            current = node.parent;
        }
        path.reverse();
        path
    }

    /// Returns the node whose segment begins at `offset` in the active
    /// transcript. Zero-length nodes are considered to start at their offset,
    /// so this prefers the deepest node when an empty branch shares a boundary
    /// with its parent.
    pub(crate) fn node_starting_at(&self, offset: usize) -> Option<ConversationNodeId> {
        let path = self.path_to(self.active);
        let mut cursor: usize = 0;
        let mut matches = Vec::new();
        for id in path {
            if cursor == offset {
                matches.push(id);
            }
            cursor = cursor.saturating_add(self.nodes.get(&id).map_or(0, |node| node.entry_count));
        }
        matches.pop()
    }

    pub fn branches(&self) -> Vec<ConversationBranch> {
        let active_path = self.path_to(self.active);
        self.nodes
            .values()
            .map(|node| ConversationBranch {
                id: node.id,
                parent: node.parent,
                label: node.label.clone(),
                entry_count: node.entry_count,
                is_active: node.id == self.active,
                is_on_active_path: active_path.contains(&node.id),
            })
            .collect()
    }

    /// Called before a user message is appended. Returns `true` when a new
    /// branch node was created, so the caller can emit a tree change event.
    pub(crate) fn begin_user_turn(&mut self, label: SharedString) -> bool {
        if let Some(node) = self.nodes.get_mut(&self.active)
            && node.awaiting_user_message
        {
            node.awaiting_user_message = false;
            node.label = label;
            return false;
        }

        let parent = self.active;
        let Ok(id) = self.create_branch(parent, label) else {
            return false;
        };
        self.active = id;
        true
    }

    /// Records that one entry was appended to the active node.
    pub(crate) fn record_entry_push(&mut self) {
        if let Some(node) = self.nodes.get_mut(&self.active) {
            node.entry_count = node.entry_count.saturating_add(1);
            node.awaiting_user_message = false;
        }
    }

    pub(crate) fn create_branch(
        &mut self,
        parent: ConversationNodeId,
        label: SharedString,
    ) -> Result<ConversationNodeId> {
        if !self.nodes.contains_key(&parent) {
            bail!("Cannot create a branch from an unknown conversation node");
        }
        let id = ConversationNodeId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.nodes.insert(
            id,
            ConversationNode {
                id,
                parent: Some(parent),
                children: Vec::new(),
                label,
                entry_count: 0,
                stored_entries: Vec::new(),
                awaiting_user_message: true,
            },
        );
        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            parent_node.children.push(id);
        }
        Ok(id)
    }

    /// Refines a label that was assigned before the user message content was
    /// known, for example when the agent echoes a keyed message in chunks.
    pub(crate) fn set_label_if_placeholder(&mut self, id: ConversationNodeId, label: SharedString) {
        if let Some(node) = self.nodes.get_mut(&id)
            && (node.label == "Branch" || node.label == "Untitled")
        {
            node.label = label;
        }
    }

    /// Moves the active transcript into `entries` for `target`, storing the
    /// previously active path back into the tree first.
    pub(crate) fn materialize(
        &mut self,
        target: ConversationNodeId,
        entries: &mut Vec<AgentThreadEntry>,
    ) -> Result<()> {
        if !self.nodes.contains_key(&target) {
            bail!("Conversation branch not found");
        }
        if target == self.active {
            return Ok(());
        }

        let previous = self.active;
        self.take_active_entries(entries);
        if self.should_prune_empty_branch(previous, target) {
            self.remove_subtree(previous);
        }

        self.active = target;
        self.load_active_path(entries);
        if let Some(node) = self.nodes.get_mut(&target)
            && node.entry_count > 0
        {
            node.awaiting_user_message = false;
        }
        Ok(())
    }

    /// Truncates the active transcript at `boundary`, optionally preserving the
    /// suffix as an inactive branch. Returns `false` when the tree and the
    /// transcript disagree, in which case the caller should fall back to a
    /// plain transcript truncation.
    pub(crate) fn truncate_active_path(
        &mut self,
        boundary: usize,
        entries: &mut Vec<AgentThreadEntry>,
        mode: ConversationTruncation,
    ) -> bool {
        let path = self.path_to(self.active);
        let total: usize = path
            .iter()
            .map(|id| self.nodes.get(id).map_or(0, |node| node.entry_count))
            .sum();
        if boundary >= total || total != entries.len() {
            return false;
        }

        self.take_active_entries(entries);

        if let Some(node_id) = self.node_starting_at(boundary) {
            // The boundary falls exactly on a turn boundary: the node keeps its
            // entries (as a sibling branch) or is removed, and its parent
            // becomes the active node.
            let parent = self.nodes.get(&node_id).and_then(|node| node.parent);
            self.active = parent.unwrap_or(self.root);
            if mode == ConversationTruncation::Discard {
                self.remove_subtree(node_id);
            }
        } else {
            // The boundary falls inside a node's own entries. Only that node's
            // suffix is truncated; deeper nodes remain archived as branches.
            let mut offset: usize = 0;
            let mut truncate_inside = None;
            for (position, id) in path.iter().enumerate() {
                let count = self.nodes.get(id).map_or(0, |node| node.entry_count);
                let end = offset.saturating_add(count);
                if boundary < end {
                    truncate_inside = Some((*id, boundary.saturating_sub(offset), position));
                    break;
                }
                offset = end;
            }
            if let Some((id, keep, position)) = truncate_inside
                && let Some(node) = self.nodes.get_mut(&id)
            {
                node.stored_entries.truncate(keep);
                node.entry_count = keep;
                self.active = if keep == 0 && position > 0 {
                    path[position - 1]
                } else {
                    id
                };
            }
        }

        self.load_active_path(entries);
        true
    }

    /// Best-effort count repair for callers that had to truncate `entries`
    /// without going through [`Self::truncate_active_path`].
    pub(crate) fn truncate_active_counts(&mut self, boundary: usize) {
        let path = self.path_to(self.active);
        let mut offset: usize = 0;
        for (position, id) in path.iter().enumerate() {
            let count = self.nodes.get(id).map_or(0, |node| node.entry_count);
            let end = offset.saturating_add(count);
            if boundary < end {
                let keep = boundary.saturating_sub(offset);
                if let Some(node) = self.nodes.get_mut(id) {
                    node.entry_count = keep;
                }
                self.active = if keep == 0 && position > 0 {
                    path[position - 1]
                } else {
                    *id
                };
                break;
            }
            if boundary == end {
                self.active = *id;
                break;
            }
            offset = end;
        }
    }

    /// Replaces the whole tree with a restored structure. Entries are not
    /// restored here; the caller replays per-node messages back into the
    /// transcript, which repopulates the nodes.
    pub(crate) fn restore(&mut self, tree: SerializedConversationTree) {
        let root = self.root;
        let mut nodes = IndexMap::default();
        nodes.insert(
            root,
            ConversationNode {
                id: root,
                parent: None,
                children: Vec::new(),
                label: "Main".into(),
                entry_count: 0,
                stored_entries: Vec::new(),
                awaiting_user_message: false,
            },
        );

        // Nodes are serialized in creation order, so a node's parent is always
        // inserted before the node itself. Anything else is dropped rather
        // than risking a disconnected subtree.
        let mut next_id = self.next_id;
        for node in tree.nodes {
            next_id = next_id.max(node.id.saturating_add(1));
            let id = ConversationNodeId::from_u64(node.id);
            if id == root {
                if let Some(root_node) = nodes.get_mut(&root) {
                    root_node.label = node.label.into();
                    root_node.awaiting_user_message = node.awaiting_user_message;
                }
                continue;
            }
            let Some(parent) = node
                .parent
                .map(ConversationNodeId::from_u64)
                .filter(|parent| nodes.contains_key(parent))
            else {
                log::warn!("dropping conversation branch with unknown parent");
                continue;
            };
            if let Some(parent_node) = nodes.get_mut(&parent) {
                parent_node.children.push(id);
            }
            nodes.insert(
                id,
                ConversationNode {
                    id,
                    parent: Some(parent),
                    children: Vec::new(),
                    label: node.label.into(),
                    entry_count: 0,
                    stored_entries: Vec::new(),
                    awaiting_user_message: node.awaiting_user_message,
                },
            );
        }

        let active = ConversationNodeId::from_u64(tree.active);
        self.nodes = nodes;
        self.active = if self.nodes.contains_key(&active) {
            active
        } else {
            self.root
        };
        self.next_id = next_id;
    }

    /// Prepares `id` to receive a replayed segment during restoration: makes
    /// it the active node and marks it as awaiting its user message, so the
    /// first replayed user message lands in the node instead of creating a
    /// child.
    pub(crate) fn begin_restore_segment(&mut self, id: ConversationNodeId) -> bool {
        if !self.nodes.contains_key(&id) {
            return false;
        }
        self.active = id;
        if let Some(node) = self.nodes.get_mut(&id) {
            node.awaiting_user_message = true;
        }
        true
    }

    /// Moves the replayed entries of the restoration segment that just ran
    /// into the active node, archiving it as an inactive branch. Unlike
    /// [`Self::materialize`], the transcript here holds only the replayed
    /// segment, so every entry belongs to the active node; splitting by the
    /// path's entry counts would misattribute them to ancestors that were
    /// restored earlier.
    pub(crate) fn store_restored_segment(&mut self, entries: &mut Vec<AgentThreadEntry>) {
        let segment = mem::take(entries);
        if let Some(node) = self.nodes.get_mut(&self.active) {
            node.entry_count = segment.len();
            node.stored_entries = segment;
        }
    }

    /// Restores the saved awaiting flag after a node's segment was replayed,
    /// since replaying an empty segment would otherwise leave the node marked
    /// as awaiting a user message it will never receive.
    pub(crate) fn set_node_awaiting_user_message(
        &mut self,
        id: ConversationNodeId,
        awaiting: bool,
    ) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.awaiting_user_message = awaiting;
        }
    }

    fn take_active_entries(&mut self, entries: &mut Vec<AgentThreadEntry>) {
        let path = self.path_to(self.active);
        let mut remaining = mem::take(entries);

        for id in path {
            let count = self
                .nodes
                .get(&id)
                .map_or(0, |node| node.entry_count)
                .min(remaining.len());
            let rest = remaining.split_off(count);
            if let Some(node) = self.nodes.get_mut(&id) {
                node.entry_count = count;
                node.stored_entries = remaining;
            }
            remaining = rest;
        }

        // If entries were appended without tree accounting, keep them attached
        // to the active node rather than dropping transcript history.
        if !remaining.is_empty()
            && let Some(node) = self.nodes.get_mut(&self.active)
        {
            node.stored_entries.append(&mut remaining);
            node.entry_count = node.stored_entries.len();
        }
    }

    fn load_active_path(&mut self, entries: &mut Vec<AgentThreadEntry>) {
        let path = self.path_to(self.active);
        entries.clear();
        for id in path {
            let Some(node) = self.nodes.get_mut(&id) else {
                continue;
            };
            let start = entries.len();
            entries.append(&mut node.stored_entries);
            node.entry_count = entries.len().saturating_sub(start);
        }
    }

    fn should_prune_empty_branch(
        &self,
        previous: ConversationNodeId,
        target: ConversationNodeId,
    ) -> bool {
        let Some(node) = self.nodes.get(&previous) else {
            return false;
        };
        previous != self.root
            && node.awaiting_user_message
            && node.entry_count == 0
            && node.stored_entries.is_empty()
            && node.children.is_empty()
            && !self.path_to(target).contains(&previous)
    }

    fn remove_subtree(&mut self, id: ConversationNodeId) {
        let mut pending = vec![id];
        while let Some(id) = pending.pop() {
            let Some(node) = self.nodes.shift_remove(&id) else {
                continue;
            };
            if let Some(parent) = node.parent
                && let Some(parent_node) = self.nodes.get_mut(&parent)
            {
                parent_node.children.retain(|child| *child != id);
            }
            if self.active == id {
                self.active = node.parent.unwrap_or(self.root);
            }
            pending.extend(node.children);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentThreadEntry, ElicitationEntryId};
    use std::sync::Arc;

    fn entry(id: &str) -> AgentThreadEntry {
        AgentThreadEntry::Elicitation(ElicitationEntryId(Arc::from(id)))
    }

    fn entry_ids(entries: &[AgentThreadEntry]) -> Vec<String> {
        entries
            .iter()
            .filter_map(|entry| match entry {
                AgentThreadEntry::Elicitation(id) => Some(id.0.to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn branches_preserve_and_restore_their_entries() {
        let mut tree = ConversationTree::new();
        let mut entries = Vec::new();

        assert!(tree.begin_user_turn("First".into()));
        entries.push(entry("a1"));
        tree.record_entry_push();
        let first = tree.active_node();

        let second = tree
            .create_branch(tree.root, "Second".into())
            .expect("branch created");
        tree.materialize(second, &mut entries)
            .expect("materialized second branch");
        assert!(entries.is_empty());

        assert!(!tree.begin_user_turn("Second".into()));
        entries.push(entry("b1"));
        tree.record_entry_push();

        tree.materialize(first, &mut entries)
            .expect("materialized first branch");
        assert_eq!(entry_ids(&entries), vec!["a1"]);

        tree.materialize(second, &mut entries)
            .expect("returned to second branch");
        assert_eq!(entry_ids(&entries), vec!["b1"]);
    }

    #[test]
    fn truncating_preserves_the_old_branch() {
        let mut tree = ConversationTree::new();
        let mut entries = Vec::new();

        tree.begin_user_turn("First".into());
        entries.push(entry("a1"));
        tree.record_entry_push();
        let first = tree.active_node();

        tree.begin_user_turn("Second".into());
        entries.push(entry("b1"));
        tree.record_entry_push();
        let second = tree.active_node();

        assert_eq!(
            tree.path_to(tree.active_node()),
            vec![tree.root, first, second]
        );
        assert!(tree.truncate_active_path(1, &mut entries, ConversationTruncation::Preserve));
        assert_eq!(entry_ids(&entries), vec!["a1"]);
        assert_eq!(tree.active_node(), first);

        // The old turn is still reachable as a sibling branch.
        tree.materialize(second, &mut entries)
            .expect("archived branch is still addressable");
        assert_eq!(entry_ids(&entries), vec!["a1", "b1"]);
    }

    #[test]
    fn truncating_with_discard_removes_the_abandoned_branch() {
        let mut tree = ConversationTree::new();
        let mut entries = Vec::new();

        tree.begin_user_turn("First".into());
        entries.push(entry("a1"));
        tree.record_entry_push();
        let first = tree.active_node();

        tree.begin_user_turn("Second".into());
        entries.push(entry("b1"));
        tree.record_entry_push();
        let second = tree.active_node();

        assert!(tree.truncate_active_path(1, &mut entries, ConversationTruncation::Discard));
        assert_eq!(entry_ids(&entries), vec!["a1"]);
        assert!(tree.node(second).is_none());
        assert!(tree.node(first).expect("first branch").children.is_empty());
    }

    #[test]
    fn node_starting_at_prefers_the_deepest_zero_length_node() {
        let mut tree = ConversationTree::new();
        let mut entries = Vec::new();

        tree.begin_user_turn("First".into());
        entries.push(entry("a1"));
        tree.record_entry_push();
        let first = tree.active_node();

        let empty = tree
            .create_branch(first, "Empty".into())
            .expect("empty branch created");
        tree.materialize(empty, &mut entries)
            .expect("activated empty branch");

        assert_eq!(tree.node_starting_at(1), Some(empty));
    }

    #[test]
    fn empty_awaiting_branches_are_pruned_when_switching_away() {
        let mut tree = ConversationTree::new();
        let mut entries = Vec::new();

        tree.begin_user_turn("First".into());
        entries.push(entry("a1"));
        tree.record_entry_push();
        let first = tree.active_node();

        let empty = tree
            .create_branch(first, "Empty".into())
            .expect("empty branch created");
        tree.materialize(empty, &mut entries)
            .expect("activated empty branch");
        tree.materialize(first, &mut entries)
            .expect("returned to first branch");

        assert!(tree.node(empty).is_none());
    }

    #[test]
    fn restore_replays_segments_into_their_nodes() {
        // Build: root -> first -> second (active); a branch point under `first`
        // with a nested branch under `sibling`, so restoration has to archive a
        // node below an ancestor that already contributed entries.
        let mut tree = ConversationTree::new();
        let mut entries = Vec::new();

        tree.begin_user_turn("First".into());
        entries.push(entry("a1"));
        tree.record_entry_push();
        let first = tree.active_node();

        tree.begin_user_turn("Second".into());
        entries.push(entry("b1"));
        tree.record_entry_push();
        let second = tree.active_node();

        let sibling = tree
            .create_branch(first, "Sibling".into())
            .expect("sibling created");
        tree.materialize(sibling, &mut entries)
            .expect("materialized sibling");
        entries.push(entry("c1"));
        tree.record_entry_push();

        let nephew = tree
            .create_branch(sibling, "Nephew".into())
            .expect("nephew created");
        tree.materialize(nephew, &mut entries)
            .expect("materialized nephew");
        entries.push(entry("d1"));
        tree.record_entry_push();
        entries.push(entry("d2"));
        tree.record_entry_push();

        tree.materialize(second, &mut entries)
            .expect("back to second");
        assert_eq!(entry_ids(&entries), vec!["a1", "b1"]);

        let structure = SerializedConversationTree {
            active: second.as_u64(),
            nodes: tree
                .iter()
                .map(|(id, node)| SerializedConversationNode {
                    id: id.as_u64(),
                    parent: node.parent().map(ConversationNodeId::as_u64),
                    label: tree
                        .label(id)
                        .map(|label| label.to_string())
                        .unwrap_or_default(),
                    awaiting_user_message: node.is_awaiting_user_message(),
                    start_client_id: None,
                })
                .collect(),
        };

        // Restore into a fresh tree, replaying segments the way the agent
        // does: inactive branches first (archiving them), active path last.
        let mut restored = ConversationTree::new();
        restored.restore(structure);
        assert_eq!(restored.node_count(), tree.node_count());
        assert_eq!(restored.active_node(), second);

        let mut entries = Vec::new();
        assert!(restored.begin_restore_segment(sibling));
        entries.push(entry("c1"));
        restored.record_entry_push();
        restored.store_restored_segment(&mut entries);
        assert!(entries.is_empty());
        assert_eq!(restored.entry_count(sibling), 1);

        assert!(restored.begin_restore_segment(nephew));
        entries.push(entry("d1"));
        restored.record_entry_push();
        entries.push(entry("d2"));
        restored.record_entry_push();
        restored.store_restored_segment(&mut entries);
        assert!(entries.is_empty());
        assert_eq!(
            restored.entry_count(nephew),
            2,
            "a nested branch must keep its entries instead of losing them \
             to an ancestor restored earlier"
        );
        assert_eq!(restored.entry_count(sibling), 1);

        for node in [restored.root, first] {
            assert!(restored.begin_restore_segment(node));
        }
        entries.push(entry("a1"));
        restored.record_entry_push();
        assert!(restored.begin_restore_segment(second));
        entries.push(entry("b1"));
        restored.record_entry_push();

        assert_eq!(restored.active_node(), second);
        assert_eq!(entry_ids(&entries), vec!["a1", "b1"]);
        assert_eq!(restored.entry_count(first), 1);
        assert_eq!(restored.entry_count(second), 1);
        assert_eq!(
            restored.label(sibling).map(|label| label.to_string()),
            Some("Sibling".to_string())
        );

        // Switching to the nested branch must materialize both archived
        // segments in path order.
        restored
            .materialize(nephew, &mut entries)
            .expect("materialized nephew");
        assert_eq!(entry_ids(&entries), vec!["a1", "c1", "d1", "d2"]);
    }
}
