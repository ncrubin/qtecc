# cluster amplitudes from/to ci coefficients
# c_i = <i| exp(t) | 0> = t_i + ...
# t_i = c_i - <i| exp(t)> + t_i

from contextlib import redirect_stdout
import io

import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


from pdaggerq.parser import contracted_strings_to_tensor_terms

def main():

    pq = pdaggerq.pq_helper("fermi")

    # T = T1 + T2 + T3 + T4
    T = ['t1', 't2', 't3', 't4']

    # exp(T) ... expanded up to fourth order ... good enough to get c4
    eT = []

    # order = 0
    eT.append([1.0, ['1']])

    # order = 1
    for my_T in T :
        eT.append([1.0, [my_T]])

    # order = 2
    for my_T1 in T :
        for my_T2 in T :
            eT.append([0.5, [my_T1, my_T2]])

    # order = 3
    for my_T1 in T :
        for my_T2 in T :
            for my_T3 in T :
                eT.append([1.0 / 6.0, [my_T1, my_T2, my_T3]])

    # order = 4
    for my_T1 in T :
        for my_T2 in T :
            for my_T3 in T :
                for my_T4 in T :
                    eT.append([1.0 / 24.0, [my_T1, my_T2, my_T3, my_T4]])

    print('')
    print('#    c(a,i) = <0|i* a e(T)|0> |0>')
    print('#    t(a,i) = c(a,i) - <0|i* a e(T)|0> |0> + t(a,i)')
    print('')

    pq.set_left_operators([['e1(i,a)']])
    pq.set_right_operators([['1']])

    pq.add_operator_product(1.0, ['r1'])
    for term in eT:
        pq.add_operator_product(-term[0], term[1])
    pq.add_operator_product(1.0, ['t1'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print(my_term)

    einsum_output = io.StringIO()
    t1 = contracted_strings_to_tensor_terms(terms)
    with redirect_stdout(einsum_output) as f:
        print('def ci_to_cc(r1, r2, r3, r4):')
        print('')
        # print('')
        for my_term in t1:
            print("#\t", my_term)
            print("%s" % (my_term.einsum_string(update_val='t1',
                                                # output_variables=('i', 'a'),
                                                )))
            print()
    
    pq.clear()

    print('')
    print('#    c(ab,ij) = <0|i* j* b a e(T)|0> |0>')
    print('#    t(ab,ij) = c(ab,ij) - <0|i* j* b a e(T)|0> |0> + t(ab,ij)')
    print('')

    pq.set_left_operators([['e2(i,j,b,a)']])
    pq.set_right_operators([['1']])

    pq.add_operator_product(1.0, ['r2'])
    for term in eT:
        pq.add_operator_product(-term[0], term[1])
    pq.add_operator_product(1.0, ['t2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print(my_term)

    t2 = contracted_strings_to_tensor_terms(terms)
    with redirect_stdout(einsum_output) as f:
        for my_term in t2:
            print("#\t", my_term)
            print("%s" % (my_term.einsum_string(update_val='t2',
                                                output_variables=('a', 'b', 'i', 'j'),
                                                )))
            print()

    pq.clear()

    print('')
    print('#    c(abc,ijk) = <0|i* j* k* c b a e(T)|0> |0>')
    print('#    t(abc,ijk) = c(abc,ijk) - <0|i* j* k* c b a e(T)|0> |0> + t(abc,ijk)')
    print('')

    pq.set_left_operators([['e3(i,j,k,c,b,a)']])
    pq.set_right_operators([['1']])

    pq.add_operator_product(1.0, ['r3'])
    for term in eT:
        pq.add_operator_product(-term[0], term[1])
    pq.add_operator_product(1.0, ['t3'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print("# ", my_term)

    t3 = contracted_strings_to_tensor_terms(terms)
    with redirect_stdout(einsum_output) as f:
        for my_term in t3:
            print("#\t", my_term)
            print("%s" % (my_term.einsum_string(update_val='t3',
                                                output_variables=('a', 'b', 'c', 'i', 'j', 'k')
                                                )))
            print()

    pq.clear()

    print('')
    print('#    c(abcd,ijkl) = <0|i* j* k* l* d c b a e(T)|0> |0>')
    print('#    t(abcd,ijkl) = c(abcd,ijkl) - <0|i* j* k* l* d c b a e(T)|0> |0> + t(abcd,ijkl)')
    print('')

    pq.set_left_operators([['e4(i,j,k,l,d,c,b,a)']])
    pq.set_right_operators([['1']])

    pq.add_operator_product(1.0, ['r4'])
    for term in eT:
        pq.add_operator_product(-term[0], term[1])
    pq.add_operator_product(1.0, ['t4'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print(my_term)

    t4 = contracted_strings_to_tensor_terms(terms)
    with redirect_stdout(einsum_output) as f:
        for my_term in t4:
            print("#\t", my_term)
            print("%s" % (my_term.einsum_string(update_val='t4',
                                                output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l'),
                                                )))
            print()

    with open("ci_to_cc.py", "w") as outf:
        outf.write(einsum_output.getvalue())
    pq.clear()

if __name__ == "__main__":
    main()

