import csv
import torch
from typing import Any
from pathlib import Path

def move_to_device(data: Any, device: torch.device) -> Any:  
    """Recursively moves all torch.Tensor objects in a nested structure to the specified device.  

    This function traverses through nested data structures such as dictionaries, lists, and tuples,  
    and moves all `torch.Tensor` objects to the specified device (e.g., 'cuda' or 'cpu'). Non-tensor  
    objects are left unchanged.  

    Args:  
        data: The input data, which can be a nested structure containing dictionaries, lists, tuples,  
              torch.Tensor objects, or other types.  
        device: The target device to which all torch.Tensor objects should be moved.  

    Returns:  
        The input data structure with all torch.Tensor objects moved to the specified device. The  
        structure of the input data (e.g., dict, list, tuple) is preserved.  
    """
    if isinstance(data, dict):  
        return {k: move_to_device(v, device) for k, v in data.items()}  
    elif isinstance(data, list):  
        return [move_to_device(v, device) for v in data]  
    elif isinstance(data, tuple):  
        return tuple(move_to_device(v, device) for v in data)  
    elif isinstance(data, torch.Tensor):  
        return data.to(device)  
    else:
        return data

def write_csv(filename : Path, head : list[Any], data : list[list[Any]]) -> None:
    """  
    Writes data to a CSV file with the specified header.  

    This function takes a file path, a list of column headers, and a 2D list of data,  
    and writes them into a CSV file. Each inner list in the 'data' parameter represents   
    a column, and data is written in column-wise order to produce a correctly formatted  
    CSV file.  

    Args:  
        filename (Path):  
            The path to the CSV file where data will be written.  
        head (list[Any]):  
            A list of column headers, each representing a column in the CSV file.   
            Length must match the number of columns in 'data'.  
        data (list[list[Any]]):  
            A 2D list where each inner list represents a column of data.   
            Each column (inner list) should have the same length.  

    Raises:  
        AssertionError:  
            If the length of 'head' does not match the number of columns in 'data'.   
            This ensures that each header corresponds to one column of data.  
    """
    assert len(head) == len(data), f"Head length {len(head)} does not match data length {len(data)}"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for i in range(len(data[0])):
            writer.writerow([data[j][i] for j in range(len(data))])