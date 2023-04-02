# https://bioinformatics.stackexchange.com/questions/19570/read-pdb-file-extract-dihedral-angles-modify-dihedral-angles-reconstruct-cart#
# ^ reference for this code 


from Bio import PDB
import math

from Bio.PDB import PDBIO
import Bio.PDB.internal_coords as ic

class ChainSelect(PDB.Select):
    def accept_chain(self, chain):
        if chain.id == 'A':
            return True
        else:
            return False

def pdb_from_dihedral_angles(structure: PDB.Structure.Structure,
                             output_path: str,
                             phi_angles, 
                             psi_angles):
    # Internally calculate dihedral angles for the structure 
    structure.atom_to_internal_coordinates()

    # Assume we care about the first chain only
    # internal_coords extends the chain class to include dihedral angles (and other info)
    chain: ic.IC_Chain = list(structure.get_chains())[0].internal_coord
    
    # Dictionary of all dihedral angles in the chain
    all_angles = chain.dihedra

    # Find the phi and psi dihedral angles
    # key[i]: ic.AtomKey
    # key[i].akl[3]: gives the atom name - //kind of confusing representation
    phi_keys = [k for k in all_angles.keys() if k[0].akl[3] == 'C' and 
                                                k[1].akl[3] == 'N' and
                                                k[2].akl[3] == 'CA' and
                                                k[3].akl[3] == 'C']
    psi_keys = [k for k in all_angles.keys() if k[0].akl[3] == 'N' and 
                                                k[1].akl[3] == 'CA' and
                                                k[2].akl[3] == 'C' and
                                                k[3].akl[3] == 'N']


    assert(len(phi_keys) == len(phi_angles))
    assert(len(psi_keys) == len(psi_angles))

    # Update phi angles
    # Note that angles are stored as degrees in ic
    phi_angles_degrees = list(map(math.degrees, phi_angles))
    for i, k in enumerate(phi_keys):
        all_angles[k].angle = phi_angles_degrees[i]

    # Update psi angles
    psi_angles_degrees = list(map(math.degrees, psi_angles))
    for i, k in enumerate(psi_keys):
        all_angles[k].angle = psi_angles_degrees[i]

    # Now that the dihedral angles have been changed, need to recompute cartesian coords              
    structure.internal_to_atom_coordinates()




    # Write PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path,  ChainSelect()) 


parser = PDB.PDBParser()
structure = parser.get_structure("1A3N", "data/1A3N.pdb")
pdb_from_dihedral_angles(structure, "output/n1A3N.pdb", [0] * 140, [0] * 140)

# structure.atom_to_internal_coordinates()

# chain = list(structure.get_chains())[0]
# ic_chain: PDB.internal_coords.IC_Chain = chain.internal_coord
# d:  Dict[Tuple[PDB.internal_coords.AtomKey, 
#                PDB.internal_coords.AtomKey,
#                PDB.internal_coords.AtomKey,
#                PDB.internal_coords.AtomKey],
#                PDB.internal_coords.Dihedron] = ic_chain.dihedra



# cnt = 1
# for key in d:

    
#     if w == 'N':
#         if key[1].akl[3] == 'CA':
#             if key[2].akl[3] == 'C':
#                 if key[3].akl[3] == 'N':
        
#                     print ('\n',cnt,' :   ',  [x.akl[3] for x in key], d[key].angle)
                    
#                     d[key].angle += 45
        
#                     cnt += 1

# cnt = 1
# for key in d:

    
#     if key[0].akl[3] == 'N':
#         if key[1].akl[3] == 'CA':
#             if key[2].akl[3] == 'C':
#                 if key[3].akl[3] == 'N':
        
#                     print ('\n',cnt,' :   ',  [x.akl[3] for x in key], d[key].angle)
        
#                     cnt += 1
                    
# structure.internal_to_atom_coordinates(verbose = True)

# io = PDBIO()

# io.set_structure(structure)

# io.save('output/atom_coord.pdb',  preserve_atom_numbering=True) 