#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#

from waterkit import Map
from waterkit import utils


ad_map = Map.from_fld('water_maps.fld')
utils.prepare_water_map(ad_map, 'tip3p')
ad_map.to_map(['OW', 'HW'], '../water')
