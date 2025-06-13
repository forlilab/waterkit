// use kdtree::{distance::squared_euclidean, KdTree};
// use rstar::{RTree, RTreeObject, AABB, PointDistance};
// use crate::atom::Atom;

// #[derive(Clone, Debug)]
// pub struct SpatialAtom {
//     index: usize,
//     coords: [f64; 3],
// }

// impl PartialEq for SpatialAtom {
//     fn eq(&self, other: &Self) -> bool {
//         self.index == other.index && self.coords == other.coords
//     }
// }

// impl Eq for SpatialAtom { }

// impl RTreeObject for SpatialAtom {
//     type Envelope = AABB<[f64; 3]>;

//     fn envelope(&self) -> Self::Envelope {
//         AABB::from_point(self.coords)
//     }
// }

// impl PointDistance for SpatialAtom {
//     fn distance_2(&self, point: &[f64; 3]) -> f64 {
//         let dx = self.coords[0] - point[0];
//         let dy = self.coords[1] - point[1];
//         let dz = self.coords[2] - point[2];
//         dx * dx + dy * dy + dz * dz
//     }
// }

// pub fn build_spatial_index(atoms: &[Atom]) -> RTree<SpatialAtom> {
//     RTree::bulk_load(
//         atoms
//             .iter()
//             .enumerate()
//             .map(|(i, atom)| SpatialAtom { 
//                 index: i, 
//                 coords: atom.coords() 
//             })
//             .collect(),
//     )
// }

// pub fn get_neighbors(
//     rtree: &RTree<SpatialAtom>,
//     water_positions: &[[f64; 3]; 3],
//     cutoff: f64,
//     exclude_indices: Option<&[usize]>
// ) -> Vec<usize> {
//     let mut neighbors = Vec::new();
//     let cutoff_sq = cutoff * cutoff;
//     let exclude_set = exclude_indices.unwrap_or(&[]);

//     for &pos in water_positions {
//         let query_point = [pos[0], pos[1], pos[2]];
//         let neighbor_points = rtree.locate_within_distance(query_point, cutoff_sq);
//         for neighbor in neighbor_points {
//             let idx = neighbor.index;
//             if !exclude_set.contains(&idx) && !neighbors.contains(&idx) {
//                 neighbors.push(idx);
//             }
//         }
//     }

//     neighbors
// }

// // pub fn build_spatial_index(atoms: &[Atom]) -> KdTree<f64, usize, [f64; 3]> {
// //     let mut tree: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);
// //     for (idx, atom) in atoms.iter().enumerate() {
// //         let _ = tree.add(atom.coords(), idx);
// //     }
// //     tree
// // }

// // pub fn get_neighbors(
// //     rtree: &KdTree<f64, usize, [f64; 3]>,
// //     water_positions: &[[f64; 3]; 3],
// //     cutoff: f64,
// // ) -> Vec<usize> {
// //     let mut neighbors = Vec::new();
// //     let cutoff_sq = cutoff * cutoff;

// //     for &pos in water_positions {
// //         let query_point = [pos[0], pos[1], pos[2]];
// //         let neighbor_points = rtree.within(&query_point, cutoff_sq, &squared_euclidean).unwrap();
// //         for neighbor in neighbor_points {
// //             if !neighbors.contains(neighbor.1) {
// //                 neighbors.push(neighbor.1.clone());
// //             }
// //         }
// //     }

// //     neighbors
// // }

// pub struct System<'a> {
//     pub atoms: &'a mut Vec<Atom>, // Borrowing a mutable reference
//     // pub rtree: KdTree<f64, usize, [f64; 3]>, // Spatial index
//     pub rtree: RTree<SpatialAtom>,
// }

// impl<'a> System<'a> {
//     pub fn new(atoms: &'a mut Vec<Atom>) -> Self {
//         let rtree = build_spatial_index(&atoms);
//         Self { atoms, rtree }
//     }

//     pub fn update_atom_position(&mut self, index: usize, new_position: [f64; 3]) {
//         // Validate new_position
//         for (i, &coord) in new_position.iter().enumerate() {
//             if !coord.is_finite() || coord.abs() > 1e6 { // Arbitrary large threshold
//                 eprintln!(
//                     "Error: Invalid coordinate at index {} in new_position: {:?} (coord {} = {})",
//                     index, new_position, i, coord
//                 );
//                 return;
//             }
//         }
//         // Update the atom's position
//         let old_coords = self.atoms[index].coords();
//         self.atoms[index].set_coords(new_position);
//         // self.rtree = build_spatial_index(self.atoms);
//         // // Remove the old atom from the tree
//         let old_atom = SpatialAtom {
//             index,
//             coords: old_coords,
//         };
//         if self.rtree.remove(&old_atom).is_none() {
//             eprintln!("Warning: Failed to remove atom at index {}", index);
//         }

//         // Insert the updated atom into the tree
//         let new_atom = SpatialAtom {
//             index,
//             coords: new_position,
//         };
//         // println!("New Positions: {:?}", new_position);
//         self.rtree.insert(new_atom);
//     }
// }

use kdtree::{distance::squared_euclidean, KdTree};
use nalgebra::Point3;
use rand::Rng;
use std::f64::consts::PI;

use crate::{atom::Atom, water::{self, WaterMolecule}};

// Atom type (water H, water O, or protein atom)
#[derive(Clone, Copy, Debug)]
pub enum AtomType {
    WaterH,
    WaterO,
    Protein,
}

// Atom with position and type
#[derive(Clone, Debug)]
pub struct AtomSystem {
    pub atom: Atom,
    atom_type: AtomType,
}

impl AtomSystem {
    pub fn new(atom: Atom, atom_type: AtomType) -> Self {
        Self { atom, atom_type }
    }

    pub fn coords(&self) -> [f64; 3] {
        self.atom.coords()
    }

    pub fn set_coords(&mut self, new_coords: [f64; 3]) {
        self.atom.set_coords(new_coords);
    }

    pub fn atom_type(&self) -> AtomType {
        self.atom_type
    }

    pub fn residue_number(&self) -> usize {
        self.atom.residue_number
    }
}

// Water molecule: indices of H1, O, H2 atoms
#[derive(Clone, Copy, Debug)]
pub struct WaterSystem {
    o_index: usize,
    h1_index: usize,
    h2_index: usize,
}

impl WaterSystem {
    pub fn new(o_index: usize, h1_index: usize, h2_index: usize) -> Self {
        Self { h1_index, o_index, h2_index }
    }

    pub fn atom_indices(&self) -> [usize; 3] {
        [self.o_index, self.h1_index, self.h2_index]
    }
}

pub struct System {
    pub atoms: Vec<AtomSystem>,
    pub waters: Vec<WaterSystem>,
    kdtree: KdTree<f64, usize, [f64; 3]>,
}

impl System {
    pub fn new(atoms: Vec<AtomSystem>, waters: Vec<WaterSystem>) -> Self {
        let kdtree = Self::build_kdtree(&atoms);
        Self { atoms, waters, kdtree }
    }

    fn build_kdtree(atoms: &[AtomSystem]) -> KdTree<f64, usize, [f64; 3]> {
        let mut kdtree = KdTree::new(3);
        for (i, atom) in atoms.iter().enumerate() {
            kdtree.add(atom.coords(), i).unwrap();
        }
        kdtree
    }

    pub fn get_neighbors(&self, center: [f64; 3], cutoff: f64, exclude_indices: &[usize]) -> Vec<usize> {
        let cutoff_sq = cutoff * cutoff;
        let neighbor_points = self.kdtree.within(&center, cutoff_sq, &squared_euclidean).unwrap();
        let mut neighbors = Vec::new();
        for neighbor in neighbor_points {
            let idx = *neighbor.1;
            if !exclude_indices.contains(&idx) && !neighbors.contains(&idx) {
                neighbors.push(idx);
            }
        }
        neighbors
    }

    pub fn update_water_position(&mut self, water_idx: usize, new_positions: [[f64; 3]; 3]) -> Result<(), String> {
        let water = self.waters[water_idx];
        let indices = water.atom_indices();

        // Update atom positions
        for (i, idx) in indices.iter().enumerate() {
            self.atoms[*idx].set_coords(new_positions[i]);
        }

        // Rebuild kdtree
        self.kdtree = Self::build_kdtree(&self.atoms);
        Ok(())
    }

    pub fn add_water_to_the_system(&mut self, water_indices: [usize; 3], water_mol: &WaterMolecule) {
        let water_vec = water_mol.as_vec();
        for atom in water_vec {
            let atom_type = match atom.atom_type().as_str() {
                "HW" => AtomType::WaterH,
                "OW" => AtomType::WaterO,
                _ => panic!("Unknown atom type: {}", atom.atom_type()),
            };
            self.atoms.push(AtomSystem::new(atom, atom_type));
            // system_atoms.push(atom.clone());
        }
        self.waters.push(WaterSystem { o_index: water_indices[0], h1_index: water_indices[1], h2_index: water_indices[2] });
        self.kdtree = Self::build_kdtree(&self.atoms);
    }

    pub fn delete_water_from_the_system(&mut self, water_indices: [usize; 3], res_number: usize) {
        self.waters.retain(|x| x.atom_indices() != water_indices);
        self.atoms.retain(|p| p.atom.residue_number != res_number);

        self.kdtree = Self::build_kdtree(&self.atoms);
    }

    pub fn water_center(&self, water_idx: usize) -> [f64; 3] {
        let water = self.waters[water_idx];
        self.atoms[water.o_index].coords()
    }

    pub fn water_positions(&self, water_idx: usize) -> [[f64; 3]; 3] {
        let water = self.waters[water_idx];
        [
            self.atoms[water.o_index].coords(),
            self.atoms[water.h1_index].coords(),
            self.atoms[water.h2_index].coords(),
        ]
    }
}