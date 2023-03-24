using BioSequences
using BioStructures

const DihedralAngle = AbstractVector{Real}

const Residue = Tuple{AminoAcid, DihedralAngle}
