from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.starcraft2.maps import smac_maps, smac_plus_maps

def get_map_params(map_name, type="smac"):
    if type == "smac_plus":
        map_param_registry = smac_plus_maps.get_smac_map_registry()    
    elif type == "smac":
        map_param_registry = smac_maps.get_smac_map_registry()
    else:
        NotImplementedError
    return map_param_registry[map_name]

def get_map_params_prev(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]
