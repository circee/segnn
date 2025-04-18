from typing import Dict, List
from .CrystalSO3Dataset import CrystalSO3Dataset
from functools import lru_cache
import torch
from torch_geometric.data import Data

class MultiDataset(CrystalSO3Dataset):
    """
    Sets up a dataset of materials to their dielectric tensor and its decomposition.
    """
    def __init__(self, molecule_data: List[dict], atom_init: Dict[str, List[float]], lmax: int, 
                 radius: float = 8, max_neighbors: int = 12, normalize: bool = True):
        super().__init__(molecule_data, atom_init, lmax, radius, max_neighbors, normalize, "Dataset/Multi/coef_stats.json") 

    @lru_cache(None)
    def get(self, idx: int) -> Data:
        x, node_attr, edge_index, edge_attr, edge_dis, pos, r_ij = super().get(idx)

        # Setup target
        WF_top, WF_bottom = self._get_WF(idx)
        
        CE = self._get_CE(idx)
        
        # period = torch.ones_like(x); group = torch.ones_like(x)
        # torch.set_printoptions(profile="full")
        # print(x)
        # for i, elem in enumerate(x):
        #     period[i], group[i] = element_number_to_period_group(x[i].item())
        

        return Data(x=torch.cat((x,pos[:,2].reshape(-1,1)),1), node_attr=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_dis=edge_dis, 
                    pos=pos, r_ij=r_ij, WF_top = WF_top, WF_bottom = WF_bottom, WF=torch.cat((WF_top.reshape(-1,1),WF_bottom.reshape(-1,1)),1),
                   y=torch.cat((WF_top.reshape(-1,1),WF_bottom.reshape(-1,1),CE.reshape(-1,1)),1))
    
    def _get_WF(self, idx: int) -> torch.Tensor:

        WF_top = torch.tensor(self.molecule_data[idx]["WF_top"])
        WF_bottom = torch.tensor(self.molecule_data[idx]["WF_bottom"])
        
        WF_top = WF_top.reshape(1, 1) ; WF_bottom = WF_bottom.reshape(1, 1)
        coef_dict = {"WF_top": WF_top,
                    "WF_bottom": WF_bottom}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["WF_top"], coef_dict["WF_bottom"]
    
    def _get_CE(self, idx: int) -> torch.Tensor:

        cleavage_energy = torch.tensor(self.molecule_data[idx]["cleavage_energy"])
        cleavage_energy = cleavage_energy.reshape(1, 1)
        coef_dict = {"cleavage_energy": cleavage_energy}
        if self.normalize:
            coef_dict = self.normalizer.normalize(coef_dict)
        return coef_dict["cleavage_energy"]
    
def element_number_to_period_group(element_number):
    """
    Converts an element number (atomic number) to its period and group.

    Args:
        element_number (int): The atomic number of the element.

    Returns:
        tuple: A tuple containing the period (int) and group (int) of the element.
               Returns None if the element number is invalid.
    """
    periodic_table_data = {
        1: (1, 1),  # Hydrogen
        2: (1, 18), # Helium
        3: (2, 1),  # Lithium
        4: (2, 2),  # Beryllium
        5: (2, 13), # Boron
        6: (2, 14), # Carbon
        7: (2, 15), # Nitrogen
        8: (2, 16), # Oxygen
        9: (2, 17), # Fluorine
        10: (2, 18),# Neon
        11: (3, 1), # Sodium
        12: (3, 2), # Magnesium
        13: (3, 13),# Aluminum
        14: (3, 14),# Silicon
        15: (3, 15),# Phosphorus
        16: (3, 16),# Sulfur
        17: (3, 17),# Chlorine
        18: (3, 18),# Argon
        19: (4, 1), # Potassium
        20: (4, 2), # Calcium
        21: (4, 3), # Scandium
        22: (4, 4), # Titanium
        23: (4, 5), # Vanadium
        24: (4, 6), # Chromium
        25: (4, 7), # Manganese
        26: (4, 8), # Iron
        27: (4, 9), # Cobalt
        28: (4, 10),# Nickel
        29: (4, 11),# Copper
        30: (4, 12),# Zinc
        31: (4, 13),# Gallium
        32: (4, 14),# Germanium
        33: (4, 15),# Arsenic
        34: (4, 16),# Selenium
        35: (4, 17),# Bromine
        36: (4, 18),# Krypton
        37: (5, 1), # Rubidium
        38: (5, 2), # Strontium
        39: (5, 3), # Yttrium
        40: (5, 4), # Zirconium
        41: (5, 5), # Niobium
        42: (5, 6), # Molybdenum
        43: (5, 7), # Technetium
        44: (5, 8), # Ruthenium
        45: (5, 9), # Rhodium
        46: (5, 10),# Palladium
        47: (5, 11),# Silver
        48: (5, 12),# Cadmium
        49: (5, 13),# Indium
        50: (5, 14),# Tin
        51: (5, 15),# Antimony
        52: (5, 16),# Tellurium
        53: (5, 17),# Iodine
        54: (5, 18),# Xenon
        55: (6, 1), # Cesium
        56: (6, 2), # Barium
        57: (6, 3), # Lanthanum
        58: (6, 3), # Cerium  (Lanthanide)
        59: (6, 3), # Praseodymium (Lanthanide)
        60: (6, 3), # Neodymium (Lanthanide)
        61: (6, 3), # Promethium (Lanthanide)
        62: (6, 3), # Samarium (Lanthanide)
        63: (6, 3), # Europium (Lanthanide)
        64: (6, 3), # Gadolinium (Lanthanide)
        65: (6, 3), # Terbium (Lanthanide)
        66: (6, 3), # Dysprosium (Lanthanide)
        67: (6, 3), # Holmium (Lanthanide)
        68: (6, 3), # Erbium (Lanthanide)
        69: (6, 3), # Thulium (Lanthanide)
        70: (6, 3), # Ytterbium (Lanthanide)
        71: (6, 3), # Lutetium (Lanthanide)
        72: (6, 4), # Hafnium
        73: (6, 5), # Tantalum
        74: (6, 6), # Tungsten
        75: (6, 7), # Rhenium
        76: (6, 8), # Osmium
        77: (6, 9), # Iridium
        78: (6, 10),# Platinum
        79: (6, 11),# Gold
        80: (6, 12),# Mercury
        81: (6, 13),# Thallium
        82: (6, 14),# Lead
        83: (6, 15),# Bismuth
        84: (6, 16),# Polonium
        85: (6, 17),# Astatine
        86: (6, 18),# Radon
        87: (7, 1), # Francium
        88: (7, 2), # Radium
        89: (7, 3), # Actinium
        90: (7, 3), # Thorium (Actinide)
        91: (7, 3), # Protactinium (Actinide)
        92: (7, 3), # Uranium (Actinide)
        93: (7, 3), # Neptunium (Actinide)
        94: (7, 3), # Plutonium (Actinide)
        95: (7, 3), # Americium (Actinide)
        96: (7, 3), # Curium (Actinide)
        97: (7, 3), # Berkelium (Actinide)
        98: (7, 3), # Californium (Actinide)
        99: (7, 3), # Einsteinium (Actinide)
        100:(7, 3), # Fermium (Actinide)
        101:(7, 3), # Mendelevium (Actinide)
        102:(7, 3), # Nobelium (Actinide)
        103:(7, 3), # Lawrencium (Actinide)
        104:(7, 4), # Rutherfordium
        105:(7, 5), # Dubnium
        106:(7, 6), # Seaborgium
        107:(7, 7), # Bohrium
        108:(7, 8), # Hassium
        109:(7, 9), # Meitnerium
        110:(7, 10),# Darmstadtium
        111:(7, 11),# Roentgenium
        112:(7, 12),# Copernicium
        113:(7, 13),# Nihonium
        114:(7, 14),# Flerovium
        115:(7, 15),# Moscovium
        116:(7, 16),# Livermorium
        117:(7, 17),# Tennessine
        118:(7, 18) # Oganesson
    }

    return periodic_table_data[element_number]