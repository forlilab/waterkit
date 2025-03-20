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
        self.index == other.index && self.coords == other.coords
    }
}

impl Eq for SpatialAtom { }

impl RTreeObject for SpatialAtom {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.coords)
    }
}

impl PointDistance for SpatialAtom {
    fn distance_2(&self, point: &[f64; 3]) -> f64 {
        let dx = self.coords[0] - point[0];
        let dy = self.coords[1] - point[1];
        let dz = self.coords[2] - point[2];
        dx * dx + dy * dy + dz * dz
    }
}

pub fn build_spatial_index(atoms: &[Atom]) -> RTree<SpatialAtom> {
    RTree::bulk_load(
        atoms
            .iter()
            .enumerate()
            .map(|(i, atom)| SpatialAtom { 
                index: i, 
                coords: atom.coords() 
            })
            .collect(),
    )
}

pub fn get_neighbors(
    rtree: &RTree<SpatialAtom>,
    water_positions: &[[f64; 3]; 3],
    cutoff: f64,
    exclude_indices: Option<&[usize]>
) -> Vec<usize> {
    let mut neighbors = Vec::new();
    let cutoff_sq = cutoff * cutoff;
    let exclude_set = exclude_indices.unwrap_or(&[]);

    for &pos in water_positions {
        let query_point = [pos[0], pos[1], pos[2]];
        let neighbor_points = rtree.locate_within_distance(query_point, cutoff_sq);
        for neighbor in neighbor_points {
            let idx = neighbor.index;
            if !exclude_set.contains(&idx) && !neighbors.contains(&idx) {
                neighbors.push(idx);
            }
        }
    }

    neighbors
}

// pub fn build_spatial_index(atoms: &[Atom]) -> KdTree<f64, usize, [f64; 3]> {
//     let mut tree: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);
//     for (idx, atom) in atoms.iter().enumerate() {
//         let _ = tree.add(atom.coords(), idx);
//     }
//     tree
// }

// pub fn get_neighbors(
//     rtree: &KdTree<f64, usize, [f64; 3]>,
//     water_positions: &[[f64; 3]; 3],
//     cutoff: f64,
// ) -> Vec<usize> {
//     let mut neighbors = Vec::new();
//     let cutoff_sq = cutoff * cutoff;

//     for &pos in water_positions {
//         let query_point = [pos[0], pos[1], pos[2]];
//         let neighbor_points = rtree.within(&query_point, cutoff_sq, &squared_euclidean).unwrap();
//         for neighbor in neighbor_points {
//             if !neighbors.contains(neighbor.1) {
//                 neighbors.push(neighbor.1.clone());
//             }
//         }
//     }

//     neighbors
// }

pub struct System<'a> {
    pub atoms: &'a mut Vec<Atom>, // Borrowing a mutable reference
    // pub rtree: KdTree<f64, usize, [f64; 3]>, // Spatial index
    pub rtree: RTree<SpatialAtom>,
}

impl<'a> System<'a> {
    pub fn new(atoms: &'a mut Vec<Atom>) -> Self {
        let rtree = build_spatial_index(&atoms);
        Self { atoms, rtree }
    }

    pub fn update_water(&mut self, base_idx: usize, water_atoms: &Vec<Atom>) {
        // Update atom positions in the Vec<Atom>
        self.atoms[base_idx].set_coords(water_atoms[0].coords());
        self.atoms[base_idx + 1].set_coords(water_atoms[1].coords());
        self.atoms[base_idx + 2].set_coords(water_atoms[2].coords());
        // Rebuild the entire R*-tree
        self.rtree = build_spatial_index(self.atoms);
    }

    // pub fn update_atom_position(&mut self, index: usize, new_position: [f64; 3]) {
    //     let old_coords = self.atoms[index].coords();
    //     println!("Updating atom {}: old_coords={:?}, new_position={:?}", index, old_coords, new_position);
    //     self.atoms[index].set_coords(new_position);
    //     let old_atom = SpatialAtom { index, coords: old_coords };
    //     if self.rtree.remove(&old_atom).is_none() {
    //         eprintln!("Failed to remove atom: index={}, coords={:?}", index, old_coords);
    //     }
    //     let new_atom = SpatialAtom { index, coords: new_position };
    //     self.rtree.insert(new_atom);
    // }
}