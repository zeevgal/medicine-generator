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
from random import sample, randint
from math import ceil

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
restricted_atom_names = ['Cl', 'F', 'O']
optional_atom_names = ['Mg', 'Be', 'Ca', 'Sr', 'Zn']
max_num_of_new_molecules = 20
min_num_of_new_molecules = 10
min_num_of_additional_atoms = 14
max_num_of_additional_atoms = 30

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
    num = 3
    if atom_id in levels[0]:
        num = 0
    else:
        if atom_id in levels[1]:
            num = 1
        else:
            if atom_id in levels[2]:
                num = 2
    return num


def get_atoms_levels(atom_num, mol_text):
    l1, l2 = get_scaffold_atom_bonds(atom_num, mol_text)
    l3 = get_childs_bonds(l2, mol_text)
    return [l1, l2, l3]


def find_scaffold6_and_8_real_id(molecule):
    for row in molecule.df.itertuples(index=False):
        if np.linalg.norm(scaffold[8] - np.asarray([row.x, row.y, row.z])) < 0.1:
            id8 = row.atom_id
        else:
            if np.linalg.norm(scaffold[6] - np.asarray([row.x, row.y, row.z])) < 0.1:
                id6 = row.atom_id
    return id8, id6


def set_legal_atom_name(atom_name):
    if atom_name in restricted_atom_names:
        indx = randint(0, len(optional_atom_names)-1)
        return optional_atom_names[indx]
    else:
        return atom_name


def get_non_scaffold_atoms_from_molecule(molecule, atoms_frequency, scaffold6_num, scaffold8_num):
    """
    returns lists of scaffold atoms and their IDs, and non-scaffold atoms
    Parameters
    ----------
    molecule

    Returns
    -------
    atoms, scaffold_ids, scaffold_atoms
    """
    atoms = [[], [], [], [], [], [], []] #level0,lev
    scaffold_ids = []
    scaffold_atoms = []
    cnt = 1
    levels8 = get_atoms_levels(scaffold8_num, molecule.mol2_text)
    levels6 = get_atoms_levels(scaffold6_num, molecule.mol2_text)
    for row in molecule.df.itertuples(index=False):
        if atom_isnt_scaffold(row.x, row.y, row.z):
            atom_name = re.sub(r'\d+', '', row.atom_name)
            atom_name = set_legal_atom_name(atom_name)
            x, y, z =  row.x, row.y, row.z
            group_index = get_atom_group(row.atom_id,levels8)
            if group_index == 3: # didn't find specific group
                group_index = get_atom_group(row.atom_id, levels6) + 3 # to distinguish between levels8 and levels6
            atoms[group_index].append([atom_name, x, y, z, atom_name])
            if atom_name in atoms_frequency.keys():
                atoms_frequency[atom_name] += 1
            else:
                atoms_frequency[atom_name] = 1
        else:
            scaffold_ids.append(row.atom_id)
            scaffold_atoms.append([row.atom_id, row.atom_name, row.x, row.y, row.z, row.atom_type ])
            cnt += 1
    return atoms, scaffold_ids, scaffold_atoms, atoms_frequency


def calc_in_percentage(atoms_frequency):
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
    molecules = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    select_bonds = False
    bonds_per_atom = {}
    atoms_frequency = {}
    for mol2 in split_multimol2('aligned.MOL2'):
        single_mol = pdmol.read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])
        scaffold8_num, scaffold6_num = find_scaffold6_and_8_real_id(single_mol)
        molecule_atoms, scaffold_atoms_ids, scaffold_atoms_list, atoms_frequency = get_non_scaffold_atoms_from_molecule(single_mol, atoms_frequency, scaffold6_num, scaffold8_num)
        for i in range(7):
            molecules[i] = pd.concat([molecules[i], pd.DataFrame(molecule_atoms[i])])

        if not select_bonds:
            select_bonds = True
            bonds_list, bonds_per_atom = get_scaffold_bonds(scaffold_atoms_ids, single_mol.mol2_text, bonds_per_atom)
            scaffold_atoms = scaffold_atoms_list
    atoms_frequency = calc_in_percentage(atoms_frequency)
    return molecules, bonds_list, scaffold_atoms, bonds_per_atom, atoms_frequency


def get_scaffold_atom_bonds(atom_num, molecule_data):
    """
    search 2 layers of atoms connected to atom_num
    Parameters
    ----------
    atom_num
    molecule_data

    Returns
    -------

    """
    parts = molecule_data.split("@<TRIPOS>BOND")
    bonds_section = parts[1].split("\n")
    childs_list = []
    childs2_list = []
    for i in range(1, len(bonds_section) // 2):
        bonds = bonds_section[i].split('\t')
        if int(bonds[2]) == atom_num:
            childs_list.append(int(bonds[3]))
        else:
            if len(childs_list) > 0: # search childs connections, but they could be sparse
                if int(bonds[2]) in childs_list:
                    childs2_list.append(int(bonds[3]))
    return childs_list, childs2_list

def get_childs_bonds(childs_list, molecule_data):
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
    for i in range(1, len(bonds_section) // 2):
        bonds = bonds_section[i].split('\t')
        if int(bonds[2]) in childs_list:
            childs2_list.append(int(bonds[3]))
    return childs2_list


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
        print(new_data.iloc[i, 0])
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


def generate_atoms(num_of_new_molecules, train_df):
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
    model.fit(train_df)
    generated_data = model.sample(num_of_new_molecules)
    return generated_data


def generate_different_atoms_groups(train_data, num_of_new_molecules):
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
        generated_data = generate_atoms(num_of_new_molecules, train_data[group_num])
        fixed_data.append(clear_atom_name(generated_data, num_of_new_molecules))
    return fixed_data

class medicine():
    """
    single new molecule class
    """
    def __init__(self, scaffold_atoms, bonds_list, bonds_per_atom, max_distance, atoms_frequency, data_from_gan, num_of_additional_atoms, data_for_level):
        self.scaffold_atoms = scaffold_atoms
        self.data_from_gan = data_from_gan
        self.bonds_per_atom = bonds_per_atom
        self.atoms_frequency = atoms_frequency
        self.data_for_level = data_for_level
        self.atom_counter = scaffold_atoms[-1][0]+1
        self.atoms_to_add = []
        self.num_of_additional_atoms = num_of_additional_atoms - len(self.data_for_level)
        self.bonds = bonds_list.copy()
        self.max_distance = max_distance
        self.molecule_depth = 2
        self.new_molecule = scaffold_atoms.copy()
        self.connected_atoms = []
        # restrictions from Hadar
        self.max_molecule_depth = 6
        self.scaffold_atoms_connected = [6,8]
        self.scaffold_num_of_connections_per_atom = [2,2]
        self.add_level_atoms_to_molecule()
        self.add_atoms_to_molecule()
        self.connect_molecule()
        self.find_unconnected_atoms()


    def fix_atom_type(self, atom_name, atom_type):
        """
        fix gan failure to generate appropriate atom_type field
        Parameters
        ----------
        atom_name
        atom_type

        Returns
        -------

        """
        if atom_name not in atom_type:
            return atom_name
        else:
            return atom_type




    def add_level_atoms_to_molecule(self):
        """
        add atoms by level keeping original structure
        Returns
        -------

        """
        for group_num in range(len(self.data_for_level)):
            index = randint(0, np.maximum(len(self.data_for_level[group_num])-1, 1))
            atom = self.data_for_level[group_num].loc[index].to_numpy()
            atom[atom_type] = self.fix_atom_type(atom[atom_name], atom[atom_type])
            fixed_row = np.insert(atom, 0, self.atom_counter) # add id number to atom
            self.new_molecule.append(fixed_row)
            self.atom_counter += 1



    def add_atoms_to_molecule(self):
        """
        add atoms with learned statistics to scaffold atoms
        Returns
        -------

        """
        step_cnt = 1
        sum = 0
        for atom in sorted(self.atoms_frequency, key=self.atoms_frequency.get,
                           reverse=False):  # get atoms by types according to train databsae statiscics
            if step_cnt < len(self.atoms_frequency.keys()):
                num = ceil(self.atoms_frequency[atom] * self.num_of_additional_atoms)
            else:
                num = abs(self.num_of_additional_atoms - sum)
            sum += num
            if len(self.data_from_gan[atom]) > 0:
                if num >= len(self.data_from_gan[atom]):
                    num = randint(0, len(self.data_from_gan[atom]))
                new_atoms_indexes = sample(range(len(self.data_from_gan[atom])), num)

                for i in range(len(new_atoms_indexes)):
                    row = self.data_from_gan[atom].iloc[new_atoms_indexes[i], :].to_numpy()
                    row[atom_type] = self.fix_atom_type(row[atom_name], row[atom_type])
                    fixed_row = np.insert(row, 0, self.atom_counter) # add id number to atom
                    self.new_molecule.append(fixed_row)
                    self.atom_counter += 1
                step_cnt += 1
        self.atoms_to_add = self.new_molecule[len(self.scaffold_atoms):]


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
        for index, atom_num in enumerate(self.scaffold_atoms_connected):
            for cnt in range(self.scaffold_num_of_connections_per_atom[index]):
                if len(self.atoms_to_add) > 0:
                    nearest_atom, nearest_atom_index = self.find_nearest_atom(self.scaffold_atoms[atom_num])
                    self.bonds.append((new_bond_num, self.scaffold_atoms[atom_num][0], nearest_atom[0], 1))
                    layer.append(nearest_atom)
                    del self.atoms_to_add[nearest_atom_index]
                    new_bond_num += 1
                    self.connected_atoms.append(nearest_atom[0])
                # after connecting to all available scaffold atoms
        if len(self.atoms_to_add) > 0:
            self.calc_connections(layer)

    def calc_connections(self, former_layer):
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
        new_bond_num = int(self.bonds[-1][0]) + 1
        self.molecule_depth += 1
        for i in range(len(former_layer)):
            if len(self.atoms_to_add) > 0:
                nearest_atom, nearest_atom_index = self.find_nearest_atom(former_layer[i])
                if nearest_atom_index < np.Inf: # real connection was found
                    self.bonds.append((new_bond_num, former_layer[i][0], nearest_atom[0], 1))
                    layer.append(nearest_atom)
                    del self.atoms_to_add[nearest_atom_index]
                    new_bond_num += 1
                    self.connected_atoms.append(nearest_atom[0])
                    self.connected_atoms.append(former_layer[i][0])
        if len(self.atoms_to_add) > 0 and self.molecule_depth <= self.max_molecule_depth:
            self.calc_connections(layer)

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
        nearest_atom_index = np.Inf
        dist = self.max_distance
        for indx in range(len(self.atoms_to_add)):  # find closest atom to one of scaffold atoms
            atoms_dist = np.linalg.norm(self.atoms_to_add[indx][2:-1] - source_atom[2:-1])
            if dist > atoms_dist and atoms_dist > 0:
                dist = atoms_dist
                nearest_atom = self.atoms_to_add[indx]
                nearest_atom_index = indx
        return nearest_atom, nearest_atom_index

    def get_bonds(self):
        return self.bonds

    def get_atoms(self):
        return self.new_molecule

    def find_unconnected_atoms(self):
        """
        delete unconnected atoms from this molecule
        Returns
        -------

        """
        unique_atoms = set(self.connected_atoms)
        for indx, row in enumerate(self.new_molecule):
            if int(row[0]) > self.scaffold_atoms[-1][0]: # no need to check scaffold atoms
                if int(row[0]) not in unique_atoms:
                    del self.new_molecule[indx]

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
        if not is_header:
            filename.write(' '.join(str(x) for x in line_element))
            filename.write("\n")
        else:
            filename.write(line_element+'\n')


def write_mol2(filename, header, atoms_lines, bonds_lines):
    with open(filename, 'w') as f:
        is_header = 1
        write_seperated_lines_to_file(f, header, is_header)
        f.write('@<TRIPOS>ATOM\n')
        is_header = 0
        write_seperated_lines_to_file(f, atoms_lines, is_header)
        f.write('@<TRIPOS>BOND\n')
        write_seperated_lines_to_file(f, bonds_lines, is_header)


def get_header(atoms_num, cnt):
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
    header.append(str(atoms_num)+"   " + str(atoms_num+1) + "     0     0     0")
    header.append("SMALL")
    header.append("USER_CHARGES")
    header.append("[2-(3-chlorophenyl)thiazol-4-yl]methanamine")
    return header


def generate_molecules(num_of_new_molecules, num_of_additional_atoms, scaffold_atoms, fixed_data, bonds_list, bonds_per_atom, atoms_frequency, data_for_level):
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
    for cnt in range(num_of_new_molecules):
        new_molecule = medicine(scaffold_atoms, bonds_list, bonds_per_atom,  atoms_max_distance, atoms_frequency, fixed_data, num_of_additional_atoms, data_for_level)#build_new_molecule(scaffold_atoms, bonds_list, bonds_per_atom, new_molecule[len(scaffold):])
        new_molecule_bonds = new_molecule.get_bonds()
        new_molecule_atoms = new_molecule.get_atoms()
        header = get_header(new_molecule.atom_counter, cnt)
        write_mol2(os.path.join('generated', 'new_molecule_{}_{}.txt').format(num_of_new_molecules, cnt), header, new_molecule_atoms, new_molecule_bonds)


def split_table_by_atoms(data_table, atoms_names):
    """
    split gan input by atom type to populate new molecule according to learned atom statistics
    Parameters
    ----------
    data_table
    atoms_names

    Returns
    -------

    """
    atoms_filtered = {}
    for atom in atoms_names:
        selected = data_table.loc[data_table['col_0'] == atom] # all the entries related to specific atom
        atoms_filtered[atom] = selected # array of dataframes hold in each entry unique dataframe per atom
    return atoms_filtered



def main():
    num_of_new_molecules = randint(min_num_of_new_molecules, max_num_of_new_molecules)
    num_of_additional_atoms = randint(min_num_of_additional_atoms, max_num_of_additional_atoms)
    real_data, bonds_list, scaffold_atoms, bonds_per_atom, atoms_frequency = prepare_data()
    fixed_data = generate_different_atoms_groups(real_data, num_of_new_molecules)
    atoms_filtered = split_table_by_atoms(fixed_data[6], atoms_frequency.keys())
    generate_molecules(num_of_new_molecules, num_of_additional_atoms, scaffold_atoms, atoms_filtered, bonds_list, bonds_per_atom, atoms_frequency, fixed_data[:-1])


if __name__ == '__main__':
    main()
