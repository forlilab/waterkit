use kdtree::{distance::squared_euclidean, KdTree};
use rstar::{RTree, RTreeObject, AABB, PointDistance};
use crate::atom::Atom;

#[derive(Clone, Debug)]
pub struct SpatialAtom {
    index: usize,
    coords: [f64; 3],
}

impl PartialEq for SpatialAtom {
    fn eq(&self, other: &Self) -> bool {
        self.coords == other.coords
    }
}

impl Eq for SpatialAtom { }


pub fn build_spatial_index(atoms: &[Atom]) -> KdTree<f64, usize, [f64; 3]> {
    let mut tree: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);
    for (idx, atom) in atoms.iter().enumerate() {
        let _ = tree.add(atom.coords(), idx);
    }
    tree
}

pub fn get_neighbors(
    rtree: &KdTree<f64, usize, [f64; 3]>,
    water_positions: &[[f64; 3]; 3],
    cutoff: f64,
) -> Vec<usize> {
    let mut neighbors = Vec::new();
    let cutoff_sq = cutoff * cutoff;

    for &pos in water_positions {
        let query_point = [pos[0], pos[1], pos[2]];
        let neighbor_points = rtree.within(&query_point, cutoff_sq, &squared_euclidean).unwrap();
        for neighbor in neighbor_points {
            if !neighbors.contains(neighbor.1) {
                neighbors.push(neighbor.1.clone());
            }
        }
    }

    neighbors
}

pub struct System<'a> {
    pub atoms: &'a mut Vec<Atom>, // Borrowing a mutable reference
    pub rtree: KdTree<f64, usize, [f64; 3]>, // Spatial index
}

impl<'a> System<'a> {
    pub fn new(atoms: &'a mut Vec<Atom>) -> Self {
        let rtree = build_spatial_index(&atoms);
        Self { atoms, rtree }
    }

    pub fn update_atom_position(&mut self, index: usize, new_position: [f64; 3]) {
        // Update the atom's position
        let old_coords = self.atoms[index].coords();
        self.atoms[index].set_coords(new_position);
        self.rtree = build_spatial_index(self.atoms);
        // // Remove the old atom from the tree
        // let old_atom = SpatialAtom {
        //     index,
        //     coords: old_coords,
        // };
        // if self.rtree.remove(&old_atom).is_none() {
        //     eprintln!("Warning: Failed to remove atom at index {}", index);
        // }

        // // Insert the updated atom into the tree
        // let new_atom = SpatialAtom {
        //     index,
        //     coords: new_position,
        // };
        // self.rtree.insert(new_atom);
    }
}