"""
gan for new molecules
by: Ze'ev Gal
"""

import os
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from biopandas.mol2 import split_multimol2
from sdv.tabular import CTGAN
import re
from random import randint, seed

seed(10)
#
# training params
#
num_of_fields = 5
nb_steps = 10000
batch_size = 64
epochs = 10
atoms_max_distance = 10
atom_name = 0
atom_type = 4
restricted_atom_names = ['Cl', 'F', 'O', 'H']
optional_atom_names = ['C', 'S', 'Br', 'I', 'P']
u_shape_atoms_types = ['C', 'N']
legal_atoms_types = u_shape_atoms_types + optional_atom_names
u_shape_molecules_all = ['ZINC54418570', 'ZINC54418918', 'ZINC54418966', 'ZINC70304164', 'ZINC19944337', 'ZINC12546573',
                     'ZINC87590125', 'ZINC23253647', 'ZINC20221711', 'ZINC70304038', 'ZINC70304040', 'ZINC19944344',
                     'ZINC87590112', 'ZINC23253563', 'ZINC87589974', 'ZINC90290472', 'ZINC90292053']
u_shape_molecules = ['ZINC87590112', 'ZINC87590125', 'ZINC87589974']
u_hexagon_shape_molecules = [ 'ZINC23253563', 'ZINC19944337']
line_shape_molecules = ['ZINC54418570', 'ZINC54418918', 'ZINC54418966']
max_num_of_new_molecules = 13
min_num_of_new_molecules = 10
min_num_of_additional_atoms = 12
max_num_of_additional_atoms = 16

scaffold = {}
scaffold[1] = np.asarray([-0.6918042917276589, 1.2000906905833681, -0.00018912104917642595])
scaffold[2] = np.asarray([0.6880925147495109, 1.2064862792986977, 0.005302683671743814])
scaffold[3] = np.asarray([1.388774355664162, -0.00015453223984703983, -0.000572666705581642])
scaffold[4] = np.asarray([0.6883606832473914, -1.207138920797627, -0.0012890949676999867])
scaffold[5] = np.asarray([-0.6922428025311852, -1.2003444952913116, -0.0019142654104198102])
scaffold[6] = np.asarray([-1.3811804594022203, 0.001060978446719496, -0.0013375355388659416])
scaffold[7] = np.asarray([2.8675006177467126, -2.220337117083852e-16, 2.199418402879527e-18])
scaffold[8] = np.asarray([5.007837319105866, 0.839583722578636, 0.001165159223780272])
scaffold[9] = np.asarray([5.512332681506209, -0.4229766819307225, 0.0009116208437578784])
scaffold[10] = np.asarray([4.002426484263403, -1.3815520647124881, 1.1926223897340549e-18])
scaffold[11] = np.asarray([3.7086677280187668, 1.0245803597748813, 0.0007203888773897177])


def atom_isnt_scaffold(x, y, z):
    """
    check if this is an atom in scaffold
    Parameters
    ----------
    x
    y
    z

    Returns
    -------

    """
    atom_xyz = np.asarray([x, y, z])
    is_scaffold = False
    for point in scaffold.values():
        if np.linalg.norm(point - atom_xyz) < 0.1:
            is_scaffold = True
    return not is_scaffold


def get_atom_group(atom_id, levels):
    """
    return the group that atom related to. group0 is the 1st layer connected to scaffold atoms.
    group1 is the layer connected to group0 etc.
    Parameters
    ----------
    atom_id
    levels

    Returns
    -------

    """
    if atom_id in levels[0]:
        num = 0
    else:
        if atom_id in levels[1]:
            num = 1
        else:
            if atom_id in levels[2]:
                num = 2
            else:
                if atom_id in levels[3]:
                    num = 3
                else:
                    if atom_id in levels[4]:
                        num = 4
                    else:
                        num = 5
    return num


def get_atoms_levels(atom_num, mol_text, scaffold_ids, H_ids):
    '''
    get the atoms in the levels of distance from scaffold atom 8
    Parameters
    ----------
    atom_num
    mol_text
    scaffold_ids

    Returns
    -------

    '''
    levels = []
    max_level = 7
    atoms_nums, _ = get_childs_bonds([atom_num], mol_text, scaffold_ids, H_ids)
    levels.append(atoms_nums)
    for index in range(1, max_level):
        atoms_nums, aromatic_exists = get_childs_bonds(levels[index-1], mol_text, scaffold_ids, H_ids)
        atoms_nums = is_uniq_atoms_nums(levels, atoms_nums)
        levels.append(atoms_nums)
    return levels , aromatic_exists


def is_uniq_atoms_nums(levels, atoms_nums):
    """
    atom in lower level shouldnt appear in lowe level
    Parameters
    ----------
    levels
    atoms_nums

    Returns
    -------

    """
    atoms_nums = list(set(atoms_nums))
    for indx in range(len(levels)):
        for num in atoms_nums:
            if num in levels[indx]:
                atoms_nums.remove(num)
    return atoms_nums

def find_scaffold8_real_id(molecule):
    '''
    get atom id for scaffold atom 8
    Parameters
    ----------
    molecule

    Returns
    -------

    '''
    for row in molecule.df.itertuples(index=False):
        if np.linalg.norm(scaffold[8] - np.asarray([row.x, row.y, row.z])) < 0.1:
            id8 = row.atom_id
    return id8


def get_scaffold_atom_ids(molecule):
    '''
    get the atom_ids of all scaffold atoms
    Parameters
    ----------
    molecule

    Returns
    -------

    '''
    scaffold_ids = []
    H_ids = []
    min_distance = 0.1
    for row in molecule.df.itertuples(index=False):
        if 'H' in row.atom_name :
            H_ids.append(row.atom_id)
        for key in scaffold:
            if np.linalg.norm(scaffold[key] - np.asarray([row.x, row.y, row.z])) < min_distance:
                scaffold_ids.append(row.atom_id)
        #if len(scaffold_ids) == len(scaffold):
            #break
    return scaffold_ids, H_ids


def set_legal_atom_name(atom_name):
    '''
    change the atom names according to rules
    Parameters
    ----------
    atom_name

    Returns
    -------

    '''
    if atom_name in restricted_atom_names:
        indx = randint(0, len(optional_atom_names)-1)
        return optional_atom_names[indx]
    else:
        return atom_name


def get_level_len(levels):
    """
    calculate number of atoms in each level
    Parameters
    ----------
    levels

    Returns
    -------

    """
    levels_len = []
    for indx in range(len(levels)):
        levels_len.append(len(levels[indx]))
    return levels_len

def get_non_scaffold_atoms_from_molecule(molecule, atoms_frequency, scaffold8_num, scaffold_ids, H_ids):
    """
    returns lists of scaffold atoms and their IDs, and non-scaffold atoms
    Parameters
    ----------
    molecule

    Returns
    -------
    atoms, scaffold_ids, scaffold_atoms
    """
    atoms = [[], [], [], [], [], []] #level0,lev
    scaffold_atoms = []
    levels8, aromatic_exists = get_atoms_levels(scaffold8_num, molecule.mol2_text, scaffold_ids, H_ids)
    levels_len = get_level_len(levels8)
    for row in molecule.df.itertuples(index=False):
        if atom_isnt_scaffold(row.x, row.y, row.z):
            atom_name = re.sub(r'\d+', '', row.atom_name)
            atom_name = set_legal_atom_name(atom_name)
            x, y, z =  row.x, row.y, row.z
            group_index = get_atom_group(row.atom_id,levels8)
            atoms[group_index].append([atom_name, x, y, z, atom_name])
        else:
            scaffold_atoms.append([row.atom_id, row.atom_name, row.x, row.y, row.z, row.atom_type])
    return atoms, scaffold_ids, scaffold_atoms, atoms_frequency, levels_len, aromatic_exists


def calc_in_percentage(atoms_frequency):
    '''
    atoms frequency by type
    Parameters
    ----------
    atoms_frequency

    Returns
    -------

    '''
    sum = 0
    for atom in atoms_frequency.keys():
        sum += atoms_frequency[atom]
    for atom in atoms_frequency.keys():
        atoms_frequency[atom] = atoms_frequency[atom] / sum
    return  atoms_frequency


def prepare_data():
    """
    pre processing the input mol2 file and getting statistics about the atoms
    Returns
    -------
    real_data - array of matrices
    """
    pdmol = PandasMol2()
    molecules = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    select_bonds = False
    bonds_per_atom = {}
    atoms_frequency = {}
    cnt = 0
    sum_levels_len = np.ones(7)
    for mol2 in split_multimol2('aligned.MOL2'):
        if mol2[0] in line_shape_molecules:
            single_mol = pdmol.read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])
            scaffold_ids, H_ids = get_scaffold_atom_ids(single_mol)
            scaffold8_num = find_scaffold8_real_id(single_mol)
            molecule_atoms, scaffold_atoms_ids, scaffold_atoms_list, atoms_frequency, levels_len, aromatic_exists = get_non_scaffold_atoms_from_molecule(single_mol, atoms_frequency, scaffold8_num, scaffold_ids, H_ids)
            sum_levels_len += levels_len
            cnt += 1
            for i in range(6):
                molecules[i] = pd.concat([molecules[i], pd.DataFrame(molecule_atoms[i])])
            if not select_bonds:
                select_bonds = True
                bonds_list, bonds_per_atom = get_scaffold_bonds(scaffold_atoms_ids, single_mol.mol2_text, bonds_per_atom)
                scaffold_atoms = scaffold_atoms_list
    atoms_frequency = calc_in_percentage(atoms_frequency)
    sum_levels_len = sum_levels_len // cnt
    for index, i in enumerate(sum_levels_len):
        if i == 0:
            sum_levels_len[index] += 1
    return molecules, bonds_list, scaffold_atoms, bonds_per_atom, atoms_frequency, sum_levels_len, aromatic_exists


def get_childs_bonds(childs_list, molecule_data, scaffold_ids, H_ids):
    """
    get 3rd or more atoms layer
    Parameters
    ----------
    childs_list
    molecule_data

    Returns
    -------

    """
    parts = molecule_data.split("@<TRIPOS>BOND")
    bonds_section = parts[1].split("\n")
    childs2_list = []
    aromatic_exists = False
    for i in range(1, len(bonds_section) // 2):
        bonds = bonds_section[i].split('\t')
        if int(bonds[2]) in childs_list and int(bonds[3]) not in scaffold_ids and int(bonds[3]) not in H_ids:
            childs2_list.append(int(bonds[3]))
        else:
            if int(bonds[3]) in childs_list and int(bonds[2]) not in scaffold_ids and int(bonds[2]) not in H_ids:
                childs2_list.append(int(bonds[2]))
        if "ar" in bonds[4]:
            aromatic_exists =True
    return childs2_list, aromatic_exists


def get_scaffold_bonds(scaffold_atoms_ids, molecule_data, bonds_per_atom):
    """
    get list of inter scaffold bonds
    Parameters
    ----------
    scaffold_atoms_ids
    molecule_data

    Returns
    -------

    """
    parts = molecule_data.split("@<TRIPOS>BOND")
    bonds_section = parts[1].split("\n")
    bonds_list = []
    cnt = 1
    for i in range(1, len(bonds_section) // 2):
        bonds = bonds_section[i].split('\t')
        if int(bonds[2]) in scaffold_atoms_ids and int(bonds[3]) in scaffold_atoms_ids:
            bond_line = bonds[2:]
            bond_line.insert(0, cnt)
            bonds_list.append(bond_line)
            cnt += 1
        else:
            if int(bonds[2]) in scaffold_atoms_ids: # get how many atoms to connect for each scaffold atom
               if int(bonds[2]) in bonds_per_atom.keys():
                   bonds_per_atom[int(bonds[2])] += 1
               else:
                   bonds_per_atom[int(bonds[2])] = 1
    return bonds_list, bonds_per_atom


def clear_atom_name(new_data, num_of_new_molecules):
    """
    delete number from atom name
    Parameters
    ----------
    new_data
    num_of_new_molecules

    Returns
    -------

    """
    for i in range(num_of_new_molecules):
        new_data.iloc[i, 0] = re.sub(r'\d+', '', new_data.iloc[i, 0])
    return new_data


def make_columns(num):
    """
    make columns names to meet ctgan needs
    Parameters
    ----------
    num

    Returns
    -------

    """
    cols_names = []
    for i in range(num):
        cols_names.append("col_" + str(i))
    return cols_names


def generate_atoms(num_of_new_molecules, train_df, num_of_atoms_in_level):
    """
    fit ctgan on table of flatten zmatrices and generate new molecules
    Parameters
    ----------
    num_of_new_molecules

    Returns
    -------
    generated zmatrices
    """
    model = CTGAN()
    train_df = make_train_set(train_df, num_of_atoms_in_level)
    total_generated_data = pd.DataFrame()
    for index in range(num_of_atoms_in_level):
        model.fit(train_df[index])
        generated_data = model.sample(num_of_new_molecules * 10)
        temp = pd.concat([total_generated_data, generated_data], axis=0)
        total_generated_data = temp.reset_index().drop('index', axis=1)
    return total_generated_data


def make_train_set(train_df, types_num):
    """
    make homogenic dataframe, i.e. similar values from si,ilar molecule coordinates.
    otherwise CTGAN results would be meaningless
    Parameters
    ----------
    train_df
    types_num

    Returns
    -------

    """
    train_set = []
    if train_df.shape[0] < 10:
        num = 10 // train_df.shape[0]
        for cnt in range(num):
            train_df = pd.concat([train_df, train_df])
    train_df = train_df.sort_values(by=['col_1'])
    train_df = train_df.reset_index().drop('index', axis=1)
    length = train_df.shape[0] // types_num - 1
    pointer = 0
    for cnt in range(types_num):
        train_set.append(train_df.loc[pointer: pointer+length])
        pointer += length+1
        if length < 10:
            tmp = pd.concat([train_set[cnt], train_set[cnt]])
            train_set[cnt] = tmp.reset_index().drop('index', axis=1)
    return train_set


def generate_different_atoms_groups(train_data, num_of_new_molecules, levels_len):
    """
    generate atoms by layers similar to the original layers.
    we want to keep the chemical structure but produce variants
    Parameters
    ----------
    train_data
    num_of_new_molecules

    Returns
    -------

    """
    fixed_data = []
    cols_names = make_columns(num_of_fields)
    for group_num in range(len(train_data)):
        train_data[group_num].columns = cols_names
        generated_data = generate_atoms(num_of_new_molecules, train_data[group_num], int(levels_len[group_num]))
        fixed_data.append(clear_atom_name(generated_data, num_of_new_molecules))
    return fixed_data

class medicine():
    """
    single new molecule class
    """
    def __init__(self, scaffold_atoms, bonds_list, bonds_per_atom, max_distance, atoms_frequency, data_from_gan,
                 num_of_additional_atoms, data_for_level, levels_len, aromatic_exists):
        self.scaffold_atoms = scaffold_atoms
        self.data_from_gan = data_from_gan
        self.bonds_per_atom = bonds_per_atom
        self.atoms_frequency = atoms_frequency
        self.data_for_level = data_for_level
        self.levels_len = levels_len
        self.aromatic_exists = aromatic_exists
        self.atom_counter = scaffold_atoms[-1][0]+1
        self.atoms_to_add = []
        self.levels_atoms = {}
        self.circle_state = 0
        self.num_of_additional_atoms = num_of_additional_atoms - len(self.data_for_level)
        self.bonds = bonds_list.copy()
        self.max_distance = max_distance
        self.curr_layer_num = 1
        self.molecule_depth = 2
        self.atom_number = 10
        self.plane_params = [ 0, 0, 0, 0]
        self.add_to_coordinate = 5
        self.new_molecule = scaffold_atoms.copy()
        self.connected_atoms = []
        # restrictions from Hadar
        self.max_molecule_depth = 6
        self.scaffold_atoms_connected = [8]
        self.scaffold_num_of_connections_per_atom = [1]
        self.add_level_atoms_to_molecule()
        self.add_atoms_to_molecule()
        self.connect_molecule()
        is_aromatic, circle_nodes = self.search_aromatic()
        if is_aromatic:
            self.get_3d_plane(self.new_molecule[circle_nodes[0]][2:-1], self.new_molecule[circle_nodes[1]][2:-1],
                              self.new_molecule[circle_nodes[2]][2:-1])
            remain = circle_nodes[3:]
            for node in remain:
                self.new_molecule[node-1][2:-1] = self.get_point_on_plane(self.new_molecule[node-1][2:-1])

    def get_3d_plane(self, p1, p2, p3):
        """
        get plane function coefficients for aromatic ring plane
        Parameters
        ----------
        p1
        p2
        p3

        Returns
        -------

        """
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        cp = np.cross([float(v1[0]),float(v1[1]), float(v1[2])], [float(v2[0]),float(v2[1]), float(v2[2])])
        self.a, self.b, self.c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        self.d = np.dot(cp, p3)
        result = self.a*p1[0]+self.b*p1[1]+self.c*p1[2]-self.d

    def get_point_on_plane(self, p1):
        """
        return atom placement on the plane defined by former atoms.
        this is for creating aromatic ring on a single plane
        """
        result = np.dot([self.a,self.b,self.c], p1)-self.d
        watchdog_cnt = 0
        accuracy = 0.1
        while abs(result) > accuracy and watchdog_cnt < 500:
            last_result = result
            index = np.random.randint(0,3)
            step = np.random.uniform(-1,1)
            p1[index] += step
            result = np.dot([self.a, self.b, self.c], p1) - self.d
            if abs(result) > abs(last_result):
                p1[index] -= 2*step
                result = np.dot([self.a, self.b, self.c], p1) - self.d
                if abs(result) > abs(last_result):
                    p1[index] += step
            watchdog_cnt += 1
        return p1

    def get_equivalent_scaffold_atom(self, num):
        '''
        returns the atom id of that scaffold atom
        Parameters
        ----------
        num

        Returns
        -------

        '''
        dist = atoms_max_distance
        for index, atom in enumerate(self.scaffold_atoms):
            cur_dist = np.linalg.norm(atom[2:-1] - scaffold[num])
            if cur_dist < dist:
                dist = cur_dist
                atom_num = index
        return atom_num

    def fix_atom_type(self, atom_name, atom_type, level_mode = False):
        """
        fix gan failure to generate appropriate atom_type field
        Parameters
        ----------
        atom_name
        atom_type

        Returns
        -------

        """
        if level_mode is True and atom_name not in u_shape_atoms_types:
            atom_name = u_shape_atoms_types[randint(0,1)]
        self.atom_number += 1
        atom = re.sub("\d+", " ", atom_name)
        if atom not in legal_atoms_types:
            index = np.random.randint(len(legal_atoms_types)-1)
            atom_name = legal_atoms_types[index]
        if atom_name not in atom_type:
            return atom_name+str(self.atom_number), atom_name
        else:
            return atom_name+str(self.atom_number), atom_type

    def add_level_atoms_to_molecule(self):
        """
        add atoms by level keeping original structure
        Returns
        -------

        """
        watchdog_cnt = 0
        level_mode = True
        c_num = 3
        n_num = 2
        for group_num in range(len(self.data_for_level)):
            start_p = 0
            for cnt in range(int(self.levels_len[group_num])):
                if int(self.levels_len[group_num]) == 1:
                    index = randint(0, np.maximum(len(self.data_for_level[group_num])-1, 1))
                else:# in case of several atoms in group each atom should be selected from its appropriate indexes range
                    length = len(self.data_for_level[group_num]) // int(self.levels_len[group_num])
                    index = randint(start_p, start_p+length)
                    start_p += length
                atom = self.data_for_level[group_num].loc[index].to_numpy()
                if index > -1:
                    if atom.shape[0] < 5:
                        for single_atom in atom:
                            c_num, n_num = self.insert_single_atom(single_atom, level_mode, c_num, n_num,
                                                                         group_num)
                    else:
                        c_num, n_num = self.insert_single_atom(atom, level_mode, c_num, n_num, group_num)


    def insert_single_atom(self, atom, level_mode, c_num, n_num, group_num):
        """
        append single atom into the first levels after atom 8 of scaffold
        Parameters
        ----------
        atom
        level_mode
        c_num
        n_num
        group_num

        Returns
        -------

        """
        atom[atom_name], atom[atom_type] = self.fix_atom_type(atom[atom_name], atom[atom_type], level_mode)
        atom[atom_name] = self.check_if_c_atom_needed(atom[atom_name], c_num, n_num,
                                                      len(self.data_for_level) - group_num)
        if 'C' in atom[atom_name]:
            c_num -= 1
        else:
            n_num -= 1
        fixed_row = np.insert(atom, 0, self.atom_counter)  # add id number to atom
        self.new_molecule.append(fixed_row)
        self.atom_counter += 1
        if group_num not in self.levels_atoms:
            self.levels_atoms[group_num] = []
        self.levels_atoms[group_num].append(fixed_row)
        return c_num, n_num

    def check_if_c_atom_needed(self, atom_name, c_num, n_num, remaining_atoms):
        """
        change atom types to fulfill the rule of 2 N and 3 C in first atoms connected to scaffold
        Parameters
        ----------
        atom_name
        c_num
        n_num
        remaining_atoms

        Returns
        -------

        """
        if c_num == 0 and n_num > 0:
            atom_name = 'N'
        elif c_num > 0 and n_num == 0:
            atom_name = 'C'
        elif remaining_atoms == 2:
            if c_num > 1:
                atom_name = 'C'
            else:
                if n_num > 1:
                    atom_name = 'N'
        elif remaining_atoms == 1:
            if n_num >= 1:
                atom_name = 'N'
            else:
                atom_name = 'C'
        return atom_name

    def add_atoms_to_molecule(self):
        """
        add atoms
        Returns
        -------

        """
        last_level_atom = self.levels_atoms[len(self.data_for_level) - 1][-1]
        one_before_last = self.levels_atoms[len(self.data_for_level) - 2][-1]
        cnt = self.atom_counter
        self.data_from_gan = self.data_from_gan.sample(frac=1)
        for index in range(0, len(self.data_from_gan) - 1):
            row = self.data_from_gan.iloc[index, :].to_numpy()
            dist = np.linalg.norm(last_level_atom[2:-1] - row[1:-1])
            if dist < 2:
                row[atom_name], row[atom_type] = self.fix_atom_type(row[atom_name], row[atom_type])
                row = self.calibrate_atoms(row, last_level_atom, one_before_last)
                fixed_row = np.insert(row, 0, self.atom_counter)  # add id number to atom
                self.new_molecule.append(fixed_row)
                self.atom_counter += 1
                last_level_atom = fixed_row
                if self.atom_counter - cnt == self.num_of_additional_atoms:
                    break
        self.atoms_to_add = self.new_molecule[len(self.scaffold_atoms) + len(self.data_for_level):]

    def calc_difs_between_atoms(self, dest, last,before):
        """
        calculate distance between atoms. diff_last is between new atom and the atom before it.
        diff_before is the distance between the new atom and the atom that is connected to the atom before it
        Parameters
        ----------
        dest
        last
        before

        Returns
        -------

        """
        diff_last = np.linalg.norm(last[2:-1] - dest[1:-1])
        diff_before = np.linalg.norm(before[2:-1] - dest[1:-1])
        result = diff_before - diff_last
        return result, diff_last

    def calibrate_atoms(self, dest_atom, last_atom, one_before_last):
        """
        adjust atoms placement to be not too far from former connected atom
        Parameters
        ----------
        dest_atom
        last_atom
        one_before_last

        Returns
        -------

        """
        result, diff_last = self.calc_difs_between_atoms(dest_atom, last_atom, one_before_last)
        watchdog_cnt = 0
        num_of_cordinates = 3
        while result < 0 and diff_last < 1 and watchdog_cnt < 500:
            last_result = result
            index = np.random.randint(0, num_of_cordinates)
            step = np.random.uniform(-1, 1)
            dest_atom[index] += step
            result, diff_last = self.calc_difs_between_atoms(dest_atom, last_atom, one_before_last)
            if last_result > result:
                dest_atom[index] -= 2*step
                result, diff_last = self.calc_difs_between_atoms(dest_atom, last_atom, one_before_last)
                if last_result > result:
                    dest_atom[index] += step
            watchdog_cnt += 1
        return dest_atom

    def set_bond_type(self, atom_label):
        """
        set bonds type according to atom type

        Parameters
        ----------
        atom_label

        Returns
        -------

        """
        atom_type = re.sub(" \d+", " ", atom_label)
        if atom_type == 'C':
            return 2
        else:
            return 1

    def connect_molecule(self):
        """
        make a new molecule from scaffold and generated additional atoms
        Parameters
        ----------

        Returns
        -------

        """
        layer = []
        new_bond_num = int(self.bonds[-1][0]) + 1
        self.start_bond = new_bond_num
        for index, atom_num in enumerate(self.scaffold_atoms_connected):
            for cnt in range(int(self.levels_len[index])):
                    real_atom_num = self.get_equivalent_scaffold_atom(atom_num)
                    if cnt == 0:
                        nearest_atom = self.levels_atoms[0][0]
                    else:
                        if len(self.levels_atoms[0]) > 1:
                            nearest_atom = self.levels_atoms[0][1]
                        else:
                            nearest_atom = self.levels_atoms[1][0]
                    bond_type = self.set_bond_type(nearest_atom[1])
                    self.bonds.append((new_bond_num, self.scaffold_atoms[real_atom_num][0], nearest_atom[0], bond_type))
                    layer.append(nearest_atom)
                    new_bond_num += 1
                    self.connected_atoms.append(nearest_atom[0])
                # after connecting to all available scaffold atoms
        index += 1
        self.pairs_num = 0
        self.added_levels_atom = 1
        if len(self.atoms_to_add) > 0:
            self.calc_connections(layer, index)

    def make_new_bond(self, new_bond_num, source, dest, bond_pair_num):
        """
        add new bond to bonds section in mol2 file
        Parameters
        ----------
        new_bond_num
        source
        dest

        Returns
        -------

        """
        if bond_pair_num > 0:
            bond_type = 2
        else:
            bond_type = 1
        self.bonds.append([new_bond_num, source, dest, bond_type])
        new_bond_num += 1
        self.connected_atoms.append(source)
        self.connected_atoms.append(dest)
        return new_bond_num

    def calc_connections(self, former_layer, index):
        """
        make connections between former layers atoms ot more atoms recursively
        Parameters
        ----------
        former_layer

        Returns
        -------
        updated bonds list
        """
        layer = []
        bond_pair = 1
        end_of_molecule = False
        new_bond_num = int(self.bonds[-1][0]) + 1
        self.molecule_depth += 1
        self.curr_layer_num += 1
        if len(former_layer) == 2 and not self.aromatic_exists and self.levels_len[index] == 1:
            layer, index = self.calc_double_connections(former_layer, index)
            self.pairs_num += 2
        else:
            for i in range(len(former_layer)):
                if index < len(self.levels_len) :
                    limit = int(self.levels_len[index])
                else:
                    bonds_limit = self.get_connections_limit(former_layer[i][1])
                    bond_pair = bonds_limit[0]
                    limit = 4
                for cnt in range(limit):
                    if len(self.atoms_to_add) > 0:
                        if self.added_levels_atom < len(self.data_for_level): # still connecting atoms from known shape
                           nearest_atom, nearest_atom_index = self.get_level_nearest_atom(index, cnt, len(former_layer))
                           self.added_levels_atom += 1
                        else:
                            nearest_atom, nearest_atom_index = self.find_nearest_atom(former_layer[i])
                            self.added_levels_atom += 1
                        if nearest_atom_index < np.Inf and nearest_atom_index > 0: # real connection was found
                            self.make_new_bond(new_bond_num, former_layer[i][0], nearest_atom[0], bond_pair)
                            bond_pair -= 1
                            layer.append(nearest_atom)
                            if self.added_levels_atom > len(self.data_for_level): # delete only non-level atoms
                                del self.atoms_to_add[nearest_atom_index]
                            new_bond_num += 1
                        else:
                            end_of_molecule = True
                    if cnt == limit-1:
                        index += 1
        if len(self.atoms_to_add) > 0 and not end_of_molecule:
            self.calc_connections(layer, index)

    def get_connections_limit(self, atom_type):
        """
        get connections number and type per atom type
        Parameters
        ----------
        atom_type

        Returns
        -------

        """
        if 'Cl' in atom_type or 'Br' in atom_type or 'F' in atom_type or 'I' in atom_type:
            return (1,3)
        elif 'C' in atom_type:
            return  (4, 0)
        elif 'N' in atom_type:
            return (3,1)
        elif 'S' in atom_type or 'O' in atom_type or 'P' in atom_type:
            return (2, 2)


    def get_level_nearest_atom(self, index, cnt, former_layer_ln):
        """
        return atom from next level
        Parameters
        ----------
        index
        cnt
        former_layer_ln

        Returns
        -------

        """
        if former_layer_ln > int(self.levels_len[index]):  # connect 2 in former layer to single atom
            nearest_atom = self.levels_atoms[index][0]
        else:
            nearest_atom = self.levels_atoms[index][cnt]
        return nearest_atom, index


    def search_aromatic(self):
        """
        check if there is aromatic ring in new molecule
        Returns
        -------

        """
        members = {}
        self.max_ring_length = 6
        ring_start = self.start_bond
        for index in range(self.start_bond, len(self.bonds)):
            for cnt in range(1, 3):
                members = self.update_members_occur(members, index, cnt)
            if index - ring_start == self.max_ring_length-1:
                is_aromatic, circle_nodes = self.set_aromatic_bonds(ring_start, members)
                if not is_aromatic:
                    ring_start += 1
                else:
                    return True, circle_nodes # only one circle except scaffold
        return False, [ring_start]

    def set_aromatic_bonds(self, ring_start, members):
        """
        set bonds type to aromatic type
        Parameters
        ----------
        ring_start
        members

        Returns
        -------

        """
        is_aromatic = False
        fix_bonds, circle_nodes = self.is_circle(ring_start, members)
        if fix_bonds:
            for index in range(ring_start, ring_start+self.max_ring_length):
                for cnt in range(1, 3):
                    if self.bonds[index][cnt] in circle_nodes and members[self.bonds[index][cnt]] > 1 and members[self.bonds[index][3-cnt]] > 1:
                        self.bonds[index][-1] = "ar"
                        is_aromatic = True
        return is_aromatic, circle_nodes

    def is_circle(self, index, members):
        """
        return aromatic ring atoms
        Parameters
        ----------
        index
        members

        Returns
        -------

        """
        temp_bonds = self.bonds[index:index+self.max_ring_length]
        first_atom = self.find_1st_atom_in_circle(members)
        next_atom = first_atom
        circle_nodes = [next_atom]
        for index in range(self.max_ring_length):
            next_atom, temp_bonds = self.find_next_neighbour(members, temp_bonds, next_atom)
            if next_atom == first_atom:
                circle_nodes.sort()
                return True, circle_nodes
            else:
                circle_nodes.append(next_atom)
        return False, circle_nodes

    def find_next_neighbour(self, members,temp_bonds,source_atom):
        """
        find adjacent atom in aromatic ring
        Parameters
        ----------
        members
        temp_bonds
        source_atom

        Returns
        -------

        """
        next_link = -1
        for index in range(len(temp_bonds)):
            if temp_bonds[index][1] == source_atom:
                next_link = temp_bonds[index][2]
            elif temp_bonds[index][2] == source_atom:
                next_link = temp_bonds[index][1]
            if next_link in members and members[next_link]>1:
                del temp_bonds[index]
                return next_link, temp_bonds
        return next_link, temp_bonds

    def find_1st_atom_in_circle(self, members):
        """ return first atom with more than 2 atoms connections"""
        start_atom = -1
        for key in members:
            if members[key] > 1:
                start_atom = key
                break
        return start_atom

    def update_members_occur(self, members, index, cnt):
        """
        update atom number of connections found in molecule
        Parameters
        ----------
        members
        index
        cnt

        Returns
        -------

        """
        if self.bonds[index][cnt] in members:
                members[self.bonds[index][cnt]] += 1
        else:
                members[self.bonds[index][cnt]] = 1
        return members

    def calc_double_connections(self, former_layer, index):
        """
        connect 2 atoms to a single atom
        Parameters
        ----------
        former_layer
        index

        Returns
        -------

        """
        layer = []
        new_bond_num = int(self.bonds[-1][0]) + 1
        self.molecule_depth += 1
        self.curr_layer_num += 1
        first_atom = former_layer[0]
        second_atom = former_layer[1]
        nearest_atom = self.levels_atoms[index]
        self.make_new_bond(new_bond_num, first_atom[0], nearest_atom[0][0], 2)
        new_bond_num += 1
        self.make_new_bond(new_bond_num, second_atom[0], nearest_atom[0][0], 2)
        layer.append(nearest_atom[0])
        self.connected_atoms.append(nearest_atom[0])
        index += 1
        return layer, index


    def find_nearest_atom(self, source_atom):
            """
            find nearest atom in target_atoms to source_atom
            Parameters
            ----------
            source_atom

            Returns
            -------

            """
            nearest_atom = None
            dist = np.Inf
            nearest_atom_index = -1
            for indx in range(len(self.atoms_to_add)):  # find closest atom to former layer atom
                atoms_dist = np.linalg.norm(self.atoms_to_add[indx][2:-1] - source_atom[2:-1])
                if dist > atoms_dist and atoms_dist > 0:
                    dist = atoms_dist
                    nearest_atom = self.atoms_to_add[indx]
                    nearest_atom_index = indx
            return nearest_atom, nearest_atom_index

    def find_nearest_2_atoms(self, first_atom, second_atom, index):
        """
        find nearest atom in target_atoms to  two atoms
        Parameters
        ----------
        first_atom
        second_atom
        index

        Returns
        -------

        """
        nearest_atom_index = 0
        nearest_atom = self.levels_atoms[index]
        self.atoms_to_add[0][2:-1] = (first_atom[2:-1]+second_atom[2:-1])/2
        return nearest_atom, nearest_atom_index

    def get_bonds(self):
        return self.bonds

    def get_atoms(self):
        return self.new_molecule


def write_seperated_lines_to_file(filename, lines, is_header = 0):
    """
    write lines to external file
    Parameters
    ----------
    filename
    lines
    is_header

    Returns
    -------

    """
    for line_element in lines:
        if is_header == 0:
            filename.write(' '.join(str(x)+"\t" for x in line_element))
            filename.write("\n")
        else:
            if is_header == 2:
                filename.write(' '.join(str(x) + "\t" for x in line_element))
                filename.write("\t1 \t <0> \t 1\n")
            else:
                filename.write(line_element+'\n')


def write_mol2(f, header, atoms_lines, bonds_lines):
    """
    write single molecule in file
    Parameters
    ----------
    f
    header
    atoms_lines
    bonds_lines

    Returns
    -------

    """
    is_header = 1
    write_seperated_lines_to_file(f, header, is_header)
    f.write('@<TRIPOS>ATOM\n')
    is_header = 2
    write_seperated_lines_to_file(f, atoms_lines, is_header)
    f.write('@<TRIPOS>BOND\n')
    write_seperated_lines_to_file(f, bonds_lines)


def get_header(atoms_num, bonds_num, cnt):
    """
    generate mol2 header
    Parameters
    ----------
    atoms_num
    cnt

    Returns
    -------

    """
    header = []
    header.append("@<TRIPOS>MOLECULE")
    header.append("TRIAL_"+str(cnt))
    header.append(str(atoms_num-1)+"   " + str(bonds_num-1) + "     0     0     0")
    header.append("SMALL")
    header.append("USER_CHARGES")
    header.append("[2-(3-chlorophenyl)thiazol-4-yl]methanamine")
    return header


def generate_molecules(num_of_new_molecules, num_of_additional_atoms, scaffold_atoms, fixed_data, bonds_list,
                       bonds_per_atom, atoms_frequency, data_for_level, levels_len, filename, aromatic_exists):
    """
    generate molecules using ctgan
    Parameters
    ----------
    num_of_new_molecules
    num_of_additional_atoms
    scaffold_atoms
    fixed_data
    bonds_list
    bonds_per_atom
    atoms_frequency
    data_for_level

    Returns
    -------

    """
    with open(filename, 'w') as f:
        for cnt in range(num_of_new_molecules):
            new_molecule = medicine(scaffold_atoms, bonds_list, bonds_per_atom,  atoms_max_distance, atoms_frequency,
                                    fixed_data, num_of_additional_atoms, data_for_level, levels_len, aromatic_exists)#build_new_molecule(scaffold_atoms, bonds_list, bonds_per_atom, new_molecule[len(scaffold):])
            new_molecule_bonds = new_molecule.get_bonds()
            new_molecule_atoms = new_molecule.get_atoms()
            header = get_header(new_molecule.atom_counter, new_molecule.bonds[-1][0]+1, cnt)
            write_mol2(f, header, new_molecule_atoms, new_molecule_bonds)


def align_scaffold_num(scaffold_atoms):
    '''
    change the atom id to start from 1
    Parameters
    ----------
    scaffold_atoms

    Returns
    -------
    convert_ids - conversion table id
    scaffold_atoms - right numbering scaffold_atoms
    '''
    convert_ids = {}
    for index in range(len(scaffold_atoms)):
        convert_ids[int(scaffold_atoms[index][0])] = index+1
        scaffold_atoms[index][0] = index+1
    return convert_ids, scaffold_atoms


def align_scaffold_bonds_nums(convert_ids, bonds_list):
    '''
    fix the atom id numbers in the bonds section
    Parameters
    ----------
    convert_ids
    bonds_list

    Returns
    -------
    fixed bonds list
    '''
    for index in range(len(bonds_list)):
        bonds_list[index][1] = convert_ids[int(bonds_list[index][1])]
        bonds_list[index][2] = convert_ids[int(bonds_list[index][2])]
    return bonds_list



def main():
    num_of_new_molecules = randint(min_num_of_new_molecules, max_num_of_new_molecules)
    num_of_additional_atoms = randint(min_num_of_additional_atoms, max_num_of_additional_atoms)
    filename = os.path.join('generated', 'new_molecules_{}.MOL2').format(num_of_additional_atoms)
    real_data, bonds_list, scaffold_atoms, bonds_per_atom, atoms_frequency, levels_len, aromatic_exists = prepare_data()
    fixed_data = generate_different_atoms_groups(real_data, num_of_new_molecules,levels_len)
    convert_ids, scaffold_atoms = align_scaffold_num(scaffold_atoms)
    bonds_list = align_scaffold_bonds_nums(convert_ids, bonds_list)
    generate_molecules(num_of_new_molecules, num_of_additional_atoms, scaffold_atoms, fixed_data[5], bonds_list, bonds_per_atom, atoms_frequency, fixed_data[:-1], levels_len, filename, aromatic_exists)


if __name__ == '__main__':
    main()
