use acp_thread::{ConversationBranch, ConversationNodeId};
use collections::HashMap;
use gpui::{Bounds, Hsla, Pixels, Point, Window, fill, point, size};

pub(crate) const BRANCH_GRAPH_NODE_WIDTH: f32 = 168.0;
pub(crate) const BRANCH_GRAPH_NODE_HEIGHT: f32 = 46.0;
pub(crate) const BRANCH_GRAPH_HORIZONTAL_GAP: f32 = 52.0;
pub(crate) const BRANCH_GRAPH_VERTICAL_GAP: f32 = 18.0;
pub(crate) const BRANCH_GRAPH_MIN_ZOOM: f32 = 0.35;
pub(crate) const BRANCH_GRAPH_MAX_ZOOM: f32 = 2.5;

pub(crate) struct BranchGraphNodeLayout {
    pub id: ConversationNodeId,
    pub x: f32,
    pub y: f32,
    pub label: gpui::SharedString,
    pub entry_count: usize,
    pub is_active: bool,
    pub is_on_active_path: bool,
}

/// Lays the conversation tree out left-to-right. Leaves consume one row each;
/// parents are centered between their first and last child, which keeps long
/// linear chains on a single row instead of spreading them diagonally.
pub(crate) fn layout_branch_graph(branches: &[ConversationBranch]) -> Vec<BranchGraphNodeLayout> {
    let mut children: HashMap<ConversationNodeId, Vec<ConversationNodeId>> = HashMap::default();
    for branch in branches {
        children.entry(branch.id).or_default();
        if let Some(parent) = branch.parent {
            children.entry(parent).or_default().push(branch.id);
        }
    }

    let Some(root) = branches
        .iter()
        .find(|branch| branch.parent.is_none())
        .map(|branch| branch.id)
    else {
        return Vec::new();
    };

    fn place(
        id: ConversationNodeId,
        depth: f32,
        next_row: &mut f32,
        children: &HashMap<ConversationNodeId, Vec<ConversationNodeId>>,
        positions: &mut HashMap<ConversationNodeId, (f32, f32)>,
    ) -> f32 {
        let child_ids = children.get(&id).cloned().unwrap_or_default();
        let row = if child_ids.is_empty() {
            let row = *next_row;
            *next_row += 1.0;
            row
        } else {
            let rows = child_ids
                .into_iter()
                .map(|child| place(child, depth + 1.0, next_row, children, positions))
                .collect::<Vec<_>>();
            (rows.first().copied().unwrap_or(0.0) + rows.last().copied().unwrap_or(0.0)) * 0.5
        };
        positions.insert(id, (depth, row));
        row
    }

    let mut positions = HashMap::default();
    let mut next_row = 0.0f32;
    place(root, 0.0, &mut next_row, &children, &mut positions);

    branches
        .iter()
        .filter_map(|branch| {
            let (depth, row) = positions.get(&branch.id).copied()?;
            Some(BranchGraphNodeLayout {
                id: branch.id,
                x: depth * (BRANCH_GRAPH_NODE_WIDTH + BRANCH_GRAPH_HORIZONTAL_GAP),
                y: row * (BRANCH_GRAPH_NODE_HEIGHT + BRANCH_GRAPH_VERTICAL_GAP),
                label: branch.label.clone(),
                entry_count: branch.entry_count,
                is_active: branch.is_active,
                is_on_active_path: branch.is_on_active_path,
            })
        })
        .collect()
}

pub(crate) fn paint_branch_edge(
    window: &mut Window,
    start: Point<Pixels>,
    end: Point<Pixels>,
    thickness: Pixels,
    color: Hsla,
) {
    let mid_x = start.x + (end.x - start.x) * 0.5;
    paint_branch_segment(window, start, point(mid_x, start.y), thickness, color);
    paint_branch_segment(
        window,
        point(mid_x, start.y),
        point(mid_x, end.y),
        thickness,
        color,
    );
    paint_branch_segment(window, point(mid_x, end.y), end, thickness, color);
    // The horizontal and vertical runs are anchored at the top-left of their
    // rectangles, so a branch leaving its parent upwards does not cover the
    // corner where those runs meet, leaving whichever sibling was painted
    // earlier visible there. Fill that corner explicitly.
    paint_branch_segment(
        window,
        point(mid_x, start.y),
        point(mid_x, start.y),
        thickness,
        color,
    );
}

fn paint_branch_segment(
    window: &mut Window,
    start: Point<Pixels>,
    end: Point<Pixels>,
    thickness: Pixels,
    color: Hsla,
) {
    let left = start.x.min(end.x);
    let top = start.y.min(end.y);
    let width = (start.x - end.x).abs().max(thickness);
    let height = (start.y - end.y).abs().max(thickness);
    window.paint_quad(fill(
        Bounds::new(point(left, top), size(width, height)),
        color,
    ));
}
