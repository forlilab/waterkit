import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd
from openmm.app import Modeller, PDBFile, ForceField

# Simulation parameters
box_size = 2.5 * unit.nanometer  # 10x10x10 Å = 1x1x1 nm
temperature = 300 * unit.kelvin
pressure = 1.0 * unit.atmosphere
time_step = 2.0 * unit.femtosecond
num_steps = 10000  # 20 ps simulation (10000 * 2 fs)
report_interval = 1000  # Report every 1000 steps
friction = 1.0 / unit.picosecond  # Langevin friction coefficient

# Create an empty modeller
modeller = Modeller(app.Topology(), [])

# Define the box vectors (cubic, 1 nm per side)
box_vectors = mm.Vec3(box_size.value_in_unit(unit.nanometer), 0, 0), \
              mm.Vec3(0, box_size.value_in_unit(unit.nanometer), 0), \
              mm.Vec3(0, 0, box_size.value_in_unit(unit.nanometer))
modeller.topology.setPeriodicBoxVectors(box_vectors)

# Load the TIP3P water model (CHARMM36)
forcefield = ForceField('amber14/tip3pfb.xml')

# Add TIP3P water molecules to the box
modeller.addSolvent(forcefield)

# Count the number of water molecules added
num_waters = sum(1 for res in modeller.topology.residues() if res.name == 'HOH')
print(f"Added {num_waters} water molecules to the box.")

# Create the system
# Step 2: Create the system
system = forcefield.createSystem(modeller.topology, 
                                 nonbondedMethod=app.PME,  # Periodic electrostatics
                                 nonbondedCutoff=1.2*unit.nanometer,  # 12 Å cutoff
                                 constraints=app.HBonds,
                                 rigidWater=True)  # Constrain H-bonds for 2 fs timestep

# Add a barostat for NPT simulation
barostat = mm.MonteCarloBarostat(pressure, temperature, 25)  # Update every 25 steps
system.addForce(barostat)

# Save initial structure
with open('water_md_sim/water_box.pdb', 'w') as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f)

new_system = forcefield.createSystem(modeller.topology, 
                                 nonbondedMethod=app.PME,  # Periodic electrostatics
                                 nonbondedCutoff=1.2*unit.nanometer,  # 12 Å cutoff
                                 rigidWater=False) 
parmed_structure = pmd.openmm.topsystem.load_topology(modeller.topology, new_system, modeller.positions)
parmed_structure.save('water_md_sim/water_box.prmtop', overwrite=True, format='amber')

# openmm_struct = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
# openmm_struct.save('water_md_sim/water_box.prmtop', overwrite=True, format='amber')

# Set up the integrator
integrator = mm.LangevinMiddleIntegrator(temperature, friction, time_step)

# Create the simulation
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# Minimize energy
print("Minimizing energy...")
simulation.minimizeEnergy(maxIterations=1000)

# Set initial velocities
simulation.context.setVelocitiesToTemperature(temperature)

# Add reporters
simulation.reporters.append(app.DCDReporter('water_md_sim/trajectory.dcd', report_interval))
simulation.reporters.append(
    app.StateDataReporter(
        'water_md_sim/simulation.log', report_interval,
        step=True, potentialEnergy=True, temperature=True, volume=True
    )
)

# Run equilibration (5000 steps = 10 ps)
# print("Equilibrating...")
# simulation.step(5000)

# Run production simulation (10000 steps = 20 ps)
print("Running production simulation...")
simulation.step(num_steps)

print("Simulation complete. Saved initial structure to 'water_box.pdb' and trajectory to 'trajectory.dcd'.")