# Much of this code was adapted from Michael Golden's code


# TODO find a source for these values
function get_ideal_bond_angle(atoms::String, residue::String)
    bond_angle = 0.0
    bond_angle_std = 0.0
    if atoms == "C-N-CA" || atoms == "CA-N-C"
        if residue == "GLY"
            bond_angle = 120.6 / 180 * pi
            bond_angle_std = 1.7  / 180 * pi
        elseif residue == "PRO"
            bond_angle = 122.6 / 180 * pi
            bond_angle_std = 5.0  / 180 * pi
        else
            bond_angle = 121.7 / 180 * pi
            bond_angle_std = 1.8  / 180 * pi
        end
    elseif atoms == "N-CA-C" || atoms == "C-CA-N"
        if residue == "GLY"
            bond_angle = 112.5 / 180 * pi
            bond_angle_std = 2.9  / 180 * pi
        elseif residue == "PRO"
            bond_angle = 111.8 / 180 * pi
            bond_angle_std = 2.5  / 180 * pi
        else
            bond_angle = 111.2 / 180 * pi
            bond_angle_std = 2.8  / 180 * pi
        end
    elseif atoms == "CA-C-N" || atoms == "N-C-CA"
        if residue == "GLY"
            bond_angle = 116.4/ 180 * pi
            bond_angle_std = 2.1  / 180 * pi
        elseif residue == "PRO"
            bond_angle = 116.9 / 180 * pi
            bond_angle_std = 1.5  / 180 * pi
        else
            bond_angle = 116.2 / 180 * pi
            bond_angle_std = 2.0  / 180 * pi
        end
    end
    return bond_angle+pi, bond_angle_std
end

function get_ideal_bond_length(atoms::String, residue::String)
    bond_length = 0.0
    bond_length_std = 0.0
    if atoms == "N-CA" || atoms == "CA-N"
        if residue == "GLY"
            bond_length = 1.451
            bond_length_std = 0.016
        elseif residue == "PRO"
            bond_length = 1.466
            bond_length_std = 0.015
        else
            bond_length = 1.458
            bond_length_std = 0.019
        end
    elseif atoms == "CA-C" || atoms == "C-CA"
        if residue == "GLY"
            bond_length = 1.516
            bond_length_std = 0.018
        else
            bond_length = 1.525
            bond_length_std = 0.021
        end
    elseif atoms == "C-N" || atoms == "N-C"
        if residue == "PRO"
            bond_length = 1.341
            bond_length_std = 0.016
        else
            bond_length = 1.329
            bond_length_std = 0.014
        end
    end
    return bond_length, bond_length_std
end

function build_chain_from_angles(sequence, phi_psi, omega, bond_angles, bond_lengths; use_input_bond_angles::Bool=false, use_input_bond_lengths::Bool=false)
    chain = Chain("A")

    residue = add_residue(chain, Residue(one_to_three[string(sequence[1])]))
    N = add_atom(residue, Atom("N", Float64[-4.308, -7.299, 1.075], element="N"))
    CA = add_atom(residue, Atom("CA", Float64[-3.695, -7.053, 2.378], element="C"))
    C = add_atom(residue, Atom("C", Float64[-4.351, -5.868, 3.097], element="C"))

    for pos=2:length(sequence)
        residue = add_residue(chain, Residue(one_to_three[string(sequence[pos])]))

        torsion_angle = phi_psi[pos-1][2] + pi
        bond_angle, bond_angle_std = get_ideal_bond_angle("CA-C-N", residue.resname)
        if use_input_bond_angles
            bond_angle = bond_angles[pos][1] + pi
        end
        ab = CA.coord - N.coord
        bc = C.coord - CA.coord
        R, R_std = get_ideal_bond_length("C-N", residue.resname)
        if use_input_bond_lengths
            R = bond_lengths[pos][1]
        end
        D = Float64[R*cos(bond_angle), R*cos(torsion_angle)*sin(bond_angle), R*sin(torsion_angle)*sin(bond_angle)]
        bchat = bc/LinearAlgebra.norm(bc)
        n = cross(ab, bchat)/norm(cross(ab, bchat))
        M = hcat(bchat,cross(n,bchat),n)
        N = add_atom(residue, Atom("N", M*D + C.coord, element="N"))

        torsion_angle = omega[pos] + pi
        bond_angle, bond_angle_std = get_ideal_bond_angle("C-N-CA", residue.resname)
        if use_input_bond_angles
            bond_angle = bond_angles[pos][2] + pi
        end
        ab = C.coord - CA.coord
        bc = N.coord - C.coord
        R, R_std = get_ideal_bond_length("N-CA", residue.resname)
        if use_input_bond_lengths
            R = bond_lengths[pos][2]
        end
        D = Float64[R*cos(bond_angle), R*cos(torsion_angle)*sin(bond_angle), R*sin(torsion_angle)*sin(bond_angle)]
        bchat = bc/LinearAlgebra.norm(bc)
        n = cross(ab, bchat)/norm(cross(ab, bchat))
        M = hcat(bchat,cross(n,bchat),n)
        CA = add_atom(residue, Atom("CA", M*D + N.coord, element="C"))

        torsion_angle = phi_psi[pos][1] + pi
        bond_angle, bond_angle_std = get_ideal_bond_angle("N-CA-C", residue.resname)
        if use_input_bond_angles
            bond_angle = bond_angles[pos][3] + pi
        end
        ab = N.coord - C.coord
        bc = CA.coord - N.coord
        R, R_std = get_ideal_bond_length("CA-C", residue.resname)
        if use_input_bond_lengths
            R = bond_lengths[pos][3]
        end
        D = Float64[R*cos(bond_angle), R*cos(torsion_angle)*sin(bond_angle), R*sin(torsion_angle)*sin(bond_angle)]
        bchat = bc/LinearAlgebra.norm(bc)
        n = cross(ab, bchat)/norm(cross(ab, bchat))
        M = hcat(bchat,cross(n,bchat),n)
        C = add_atom(residue, Atom("C", M*D + CA.coord, element="C"))
    end
    return chain
end
