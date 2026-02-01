"""GPAW PAW setup file loader

This module provides functions to load GPAW PAW setup files (.gz format)
following a similar structure to jrystal's UPF loader.

Author: Claude
Date: 2025-09-03
"""

import gzip
import os
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path


def parse_radial_grid(radial_grid_element):
    """Parse the radial grid section."""
    grid_info = {
        'eq': radial_grid_element.get('eq'),
        'a': float(radial_grid_element.get('a')) if radial_grid_element.get('a') else None,
        'n': int(radial_grid_element.get('n')),
        'istart': int(radial_grid_element.get('istart')),
        'iend': int(radial_grid_element.get('iend')),
        'id': radial_grid_element.get('id')
    }
    
    # Add optional attributes if they exist
    if radial_grid_element.get('b'):
        grid_info['b'] = float(radial_grid_element.get('b'))
    if radial_grid_element.get('d'):
        grid_info['d'] = float(radial_grid_element.get('d'))
    if radial_grid_element.get('ri'):
        grid_info['ri'] = float(radial_grid_element.get('ri'))
    
    return grid_info


def parse_shape_function(shape_function_element):
    """Parse the shape function section."""
    return {
        'type': shape_function_element.get('type'),
        'rc': float(shape_function_element.get('rc'))
    }


def parse_valence_states(valence_states_element):
    """Parse the valence states section."""
    states = []
    for state in valence_states_element.findall('state'):
        state_dict = {}
        
        # Handle optional n attribute (might be missing for virtual states)
        if state.get('n'):
            state_dict['n'] = int(state.get('n'))
        
        state_dict['l'] = int(state.get('l'))
        
        # Handle f (occupation) attribute - use get with default
        if state.get('f'):
            state_dict['f'] = float(state.get('f'))
        
        if state.get('id'):
            state_dict['id'] = state.get('id')
        if state.get('rc'):
            state_dict['rc'] = float(state.get('rc'))
        if state.get('e'):  # Note: attribute is 'e' not 'energy'
            state_dict['e'] = float(state.get('e'))
        
        states.append(state_dict)
    return states


def parse_ae_core_density(element):
    """Parse the all-electron core density."""
    if element is not None:
        return list(map(float, element.text.split()))
    return None


def parse_pseudo_core_density(element):
    """Parse the pseudo core density."""
    if element is not None:
        return list(map(float, element.text.split()))
    return None


def parse_local_potential(element):
    """Parse the local potential."""
    if element is not None:
        return list(map(float, element.text.split()))
    return None


def parse_partial_waves(partial_waves_element):
    """Parse the partial waves section."""
    partial_waves = []
    for state in partial_waves_element.findall('state'):
        state_data = {
            'n': int(state.get('n')),
            'l': int(state.get('l'))
        }
        if state.get('id'):
            state_data['id'] = state.get('id')
        if state.get('rc'):
            state_data['rc'] = float(state.get('rc'))
        if state.get('energy'):
            state_data['energy'] = float(state.get('energy'))
        
        # Parse ae (all-electron) and ps (pseudo) wavefunctions
        ae_element = state.find('ae')
        if ae_element is not None:
            state_data['ae'] = list(map(float, ae_element.text.split()))
        
        ps_element = state.find('ps')
        if ps_element is not None:
            state_data['ps'] = list(map(float, ps_element.text.split()))
        
        partial_waves.append(state_data)
    return partial_waves


def parse_projectors(projectors_element):
    """Parse the projector functions section."""
    projectors = []
    for state in projectors_element.findall('state'):
        state_data = {
            'n': int(state.get('n')),
            'l': int(state.get('l'))
        }
        if state.get('id'):
            state_data['id'] = state.get('id')
        if state.get('energy'):
            state_data['energy'] = float(state.get('energy'))
        
        # Parse projector function
        projector_element = state.find('projector_function')
        if projector_element is not None:
            state_data['projector'] = list(map(float, projector_element.text.split()))
        
        projectors.append(state_data)
    return projectors


def parse_zero_potential(element):
    """Parse the zero potential."""
    if element is not None:
        return list(map(float, element.text.split()))
    return None


def parse_kresse_joubert_projectors(element):
    """Parse Kresse-Joubert projectors if present."""
    if element is None:
        return None
    
    projectors = []
    for state in element.findall('state'):
        state_data = {
            'n': int(state.get('n')),
            'l': int(state.get('l'))
        }
        if state.get('id'):
            state_data['id'] = state.get('id')
        if state.get('energy'):
            state_data['energy'] = float(state.get('energy'))
        
        # Parse projector function
        projector_element = state.find('projector_function')
        if projector_element is not None:
            state_data['projector'] = list(map(float, projector_element.text.split()))
        
        projectors.append(state_data)
    return projectors


def parse_paw_setup(filepath):
    """Parse a GPAW PAW setup file.
    
    Args:
        filepath: Path to GPAW setup file (.gz or .xml)
        
    Returns:
        dict: All parsed data from the setup file
    """
    # Read file
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt') as f:
            content = f.read()
    else:
        with open(filepath, 'r') as f:
            content = f.read()
    
    # Parse XML
    root = ET.fromstring(content.lstrip())
    result = {}
    
    # Helper to parse float values
    def parse_values(text):
        return list(map(float, text.split())) if text else []
    
    # Parse elements with attributes only
    for tag in ['atom', 'xc_functional', 'ae_energy', 'core_energy', 'shape_function']:
        elem = root.find(tag)
        if elem is not None:
            result[tag] = {}
            for k, v in elem.attrib.items():
                try:
                    result[tag][k] = float(v)
                except ValueError:
                    result[tag][k] = v
    
    # Parse radial_grid
    radial_grid = root.find('radial_grid')
    if radial_grid is not None:
        result['radial_grid'] = parse_radial_grid(radial_grid)
    
    # Parse valence_states
    valence_states = root.find('valence_states')
    if valence_states is not None:
        result['valence_states'] = parse_valence_states(valence_states)
    
    # Parse elements with text content
    for tag in ['ae_core_density', 'pseudo_core_density', 'pseudo_valence_density',
                'zero_potential', 'localpotential', 'kinetic_energy_differences']:
        elem = root.find(tag)
        if elem is not None and elem.text:
            result[tag.replace('localpotential', 'local_potential')] = parse_values(elem.text)
    
    # Parse PAW elements (direct children of root)
    for tag, key in [('ae_partial_wave', 'ae_partial_waves'),
                     ('pseudo_partial_wave', 'pseudo_partial_waves'),
                     ('projector_function', 'projector_functions')]:
        elements = root.findall(tag)
        if elements:
            result[key] = []
            for elem in elements:
                data = {
                    'state': elem.get('state'),
                    'grid': elem.get('grid'),
                    'values': parse_values(elem.text) if elem.text else []
                }
                result[key].append(data)
    
    # Parse nested containers
    for container_tag in ['projectors', 'partial_waves', 'kresse_joubert_projectors']:
        container = root.find(container_tag)
        if container is not None:
            result[container_tag] = parse_projectors(container) if 'projector' in container_tag else parse_partial_waves(container)
    
    # Parse exact_exchange
    exact_exchange = root.find('exact_exchange')
    if exact_exchange is not None:
        result['exact_exchange'] = {}
        if exact_exchange.get('ExxC'):
            result['exact_exchange']['ExxC'] = float(exact_exchange.get('ExxC'))
        X_p = exact_exchange.find('X_p')
        if X_p is not None and X_p.text:
            result['exact_exchange']['X_p'] = parse_values(X_p.text)
    
    return result


def find_gpaw_setup(dir_path, atom, xc='PBE'):
    """Find the GPAW setup file for an atom in the directory path.
    
    The setup file must follow the pattern:
        "$atom_symbol$.$xc_functional$.gz" or "$atom_symbol$.$xc_functional$"
    
    Args:
        dir_path (str): The directory path for GPAW setup files.
        atom (str): The symbol for the element.
        xc (str): The exchange-correlation functional (default: 'PBE').
        
    Returns:
        str: The setup file path.
        
    Raises:
        ValueError: If the setup file is not found.
    """
    # Ensure directory path ends with separator
    if not dir_path.endswith(os.sep):
        dir_path += os.sep
    
    # Look for both compressed and uncompressed files
    patterns = [
        f"{atom}.{xc}.gz",
        f"{atom}.{xc}"
    ]
    
    for pattern in patterns:
        filepath = os.path.join(dir_path, pattern)
        if os.path.exists(filepath):
            return filepath
    
    raise ValueError(
        f"The GPAW setup file for element '{atom}' with functional '{xc}' "
        f"is not found in directory {dir_path}"
    )