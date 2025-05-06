import pandas as pd
from dataclasses import dataclass
from collections import defaultdict 

from path import Paths
from config import Yeo_Network

# Brain network
@dataclass(frozen=True)
class Yeo_Network: # for Brainnetome Atlas
    # in file: subregion_func_network_Yeo_updated.csv
    #  7: Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default
    # 17: Visual peripheral/central, Somato-motor A/B, Dorsal attention A/B, Ventral attention, Salience, Limbic-1/2, Control C/A/B, Default D (Auditory)/C/A/B
    granularity: str = "coarse" # coarse=7 fine=17
    network: str = "Whole"  # from the above names or Whole
    select: bool = True # True=the subregions within the network; False=the complement in the whole brain.

# Networks of Brainnetome Atlas
def get_yeo_network_of_brainnetome() -> list[int] | None:
    """  
    Returns a list of subregion labels associated with or excluding a specified Yeo functional brain network.  

    This function reads a CSV file containing the mapping between subregions (from the Brainnetome atlas)  
    and functional networks (from the Yeo 7-network or 17-network parcellations). The function selects  
    the relevant network based on the specified granularity ('coarse' for 7 networks, 'fine' for 17 networks)  
    and either returns the labels of subregions that belong to the selected network or excludes them,  
    based on a user-specified flag.  

    Args:  
        None. Function relies on external configuration:  
            - Paths.Atlas.Brainnetome.subregion_func_network_Yeo_updated_csv_path: Path to the relevant CSV file.  
            - Yeo_Network.granularity: String, either "coarse" (7-network) or "fine" (17-network).  
            - Yeo_Network.network: String, the name of the Yeo network to include or exclude.  
            - Yeo_Network.select: Boolean, whether to select subregions in the specified network (True) or out of the network (False).

    Returns:  
        list[int]: Sorted list of integer labels for the selected (or deselected) subregions,  
                   or None if "Whole" network is specified.  

    Raises:  
        AssertionError: If the network granularity is unknown or the network name is not valid.  
    """ 
    if Yeo_Network.network == "Whole": # the whole functional connection matrix is selected
        return None
    
    # functional networks, Yeo
    pd_frame = pd.read_csv(Paths.Atlas.Brainnetome.subregion_func_network_Yeo_updated_csv_path, sep=",", header=1)
   
    # subregion_func_network_Yeo_updated
    subregions = pd_frame.iloc[:, :5]
    fields = ["Label", "subregion_name", "region"]
    network_name = "Yeo_7network" if Yeo_Network.granularity == "coarse" else "Yeo_17network" if Yeo_Network.granularity == "fine" else None
    assert network_name is not None, f"Unknown network name: {Yeo_Network.granularity}"
    subregion_dict = defaultdict(list)
    for _, row in subregions.iterrows():
        subregion_dict[row[network_name]].append({field: row[field] for field in fields})
   
    # Yeo 7 Network  &  Yeo 17 Network, dict={ID : Network name}
    yeo_network = pd_frame.iloc[2:9, 10:12] if Yeo_Network.granularity == "coarse" else pd_frame.iloc[12:29, 10:12] if Yeo_Network.granularity == "fine" else None
    assert yeo_network is not None, f"Unknown network name: {Yeo_Network.granularity}"
    yeo_dict  = {int(row.iloc[0]) : row.iloc[1] for _, row in yeo_network.iterrows()}
    assert Yeo_Network.network in yeo_dict.values() , f"Unknown network name: {Yeo_Network.network}, it must be in {yeo_dict.values()}"
   
    # replace the key in subregion_dict
    new_dict = {yeo_dict[k]: v for k, v in subregion_dict.items() if k in yeo_dict}
    subregion_dict.clear()  
    subregion_dict.update(new_dict)   
    if Yeo_Network.select:
        labels = sorted([item["Label"] for item in subregion_dict[Yeo_Network.network]])
    else:
        labels = sorted(list(set(pd_frame["Label"].tolist()) - set([item["Label"] for item in subregion_dict[Yeo_Network.network]])))
    
    return labels


    # self.labels = get_yeo_network_of_brainnetome()

    # def __subgraph__(self, input_array : np.ndarray) -> np.ndarray:
    #     """
    #     only for fMRI
    #     whole graph = whole brain = whole matrix, subgraph = a brain network = sub matrix
    #     """
    #     if self.labels is None:
    #         output_array = input_array
    #     else: # set (i, j) to 0
    #         labels = np.unique(self.labels)
    #         output_array = input_array.copy()
    #         output_array[np.ix_(labels, labels)] = 0 
    #     return output_array