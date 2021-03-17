#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#

import os
import glob

from vina import Vina


v = Vina()
v.set_receptor('water.pdbqt')
v.compute_vina_maps([0, 0, 0], [15, 15, 15])
v.write_maps('../water', overwrite=True)

os.rename('../water.O_DA.map', '../water_SW.map')
maps_to_delete = glob.glob('../water.*.map')

for map_to_delete in maps_to_delete:
    os.remove(map_to_delete)
