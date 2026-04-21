from pymatgen.core import Structure
from pymatgen.analysis.dimensionality import get_structure_components

# Transition and Rare Earth Elements
DF_BLOCK = {
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd",
    "Tb","Dy","Ho","Er","Tm","Yb","Lu"
}

# Covalent and Main Group Elements
ANGULAR = {
    "Si","Ge","Ga","As","In","P","Sb","Se","Te","S",
    "Bi","Sn","Pb","B","N","C","H","Cl","Br","I","F"
}

def route_crystal(cif_path):
    """
    Analyzes CIF structure and routes to the best GNN model.
    Prioritizes OGCNN for transition metals and ALIGNN for low-D or covalent systems.
    """
    struct = Structure.from_file(cif_path)
    elements = {str(site.specie.symbol) for site in struct}
    n_sites = len(struct)

    # 1. Dimensionality Check (3D vs 2D/1D)
    try:
        components = get_structure_components(struct)
        max_dim = max([c['dimensionality'] for c in components]) if components else 3
    except:
        max_dim = 3 

    has_df = bool(elements & DF_BLOCK)
    has_angular = bool(elements & ANGULAR)



    # --- RULE 1: OGCNN (Transition Metals / Specialized d-block) ---
    # Optimized for 3D d/f block systems of manageable size
    if has_df and max_dim == 3 and n_sites <= 80:
        return "OGCNN"

    # --- RULE 2: ALIGNN (Low-D or Purely Covalent) ---
    # Dimensionality is the primary signal for ALIGNN
    elif max_dim <= 2:
        return "ALIGNN"
    
    # 3D but purely covalent (No d/f block metals)
    elif has_angular and not has_df and n_sites <= 50:
        return "ALIGNN"

    
    # if max_dim <= 2:
    #     return "ALIGNN"

    # # RULE 2: ALIGNN — purely covalent 3D structures (no d/f block)
    # elif has_angular and not has_df and n_sites <= 50:
    #     return "ALIGNN"

    # # RULE 3: OGCNN — 3D transition/rare-earth metal systems
    # elif has_df and max_dim == 3 and n_sites <= 80:
    #     return "OGCNN"
    # --- RULE 3: iCGCNN (Generalist / Large / Mixed / Ionic) ---
    else:
        # Catch-all for large transition metal systems, mixed crystals, 
        # and simple ionic structures.
        return "iCGCNN"