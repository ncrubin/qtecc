from numpy.linalg import einsum


def ci_to_cc(r1, r2, r3, r4):
    #	  1.0000 r1(a,i)
    t1 +=  1.000000000000000 * einsum('ai', r1)
    
    #	  1.0000 r2(a,b,i,j)
    t2 +=  1.000000000000000 * einsum('abij->abij', r2)
    
    #	  1.0000 P(i,j)t1(a,j)*t1(b,i)
    contracted_intermediate =  1.000000000000000 * einsum('aj,bi->abij', t1, t1)
    t2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 r3(a,b,c,i,j,k)
    t3 +=  1.000000000000000 * einsum('abcijk->abcijk', r3[:, :, :, :, :, :])
    
    #	 -1.0000 P(j,k)*P(a,b)t1(a,k)*t2(b,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('ak,bcij->abcijk', t1, t2)
    t3 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)t1(a,i)*t2(b,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ai,bcjk->abcijk', t1, t2)
    t3 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)t1(c,k)*t2(a,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('ck,abij->abcijk', t1, t2)
    t3 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 t1(c,i)*t2(a,b,j,k)
    t3 += -1.000000000000000 * einsum('ci,abjk->abcijk', t1, t2)
    
    #	  1.0000 P(i,j)*P(a,b)t1(a,k)*t1(b,j)*t1(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('ak,bj,ci->abcijk', t1, t1, t1, optimize=['einsum_path', (0, 1, 2)])
    t3 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  1.0000 P(j,k)t1(a,i)*t1(b,k)*t1(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('ai,bk,cj->abcijk', t1, t1, t1, optimize=['einsum_path', (0, 1, 2)])
    t3 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 r4(a,b,c,d,i,j,k,l)
    t4 +=  1.000000000000010 * einsum('abcdijkl->abcdijkl', r4[:, :, :, :, :, :, :, :])
    
    #	  1.0000 P(k,l)*P(a,b)t1(a,l)*t3(b,c,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('al,bcdijk->abcdijkl', t1, t3)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)t1(a,j)*t3(b,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('aj,bcdikl->abcdijkl', t1, t3)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)t1(c,l)*t3(a,b,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('cl,abdijk->abcdijkl', t1, t3)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)t1(c,j)*t3(a,b,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('cj,abdikl->abcdijkl', t1, t3)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)t2(a,b,k,l)*t2(c,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abkl,cdij->abcdijkl', t2, t2)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)t2(a,b,i,l)*t2(c,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abil,cdjk->abcdijkl', t2, t2)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)t2(a,b,j,k)*t2(c,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('abjk,cdil->abcdijkl', t2, t2)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)t2(a,d,k,l)*t2(b,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('adkl,bcij->abcdijkl', t2, t2)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)t2(a,d,i,l)*t2(b,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adil,bcjk->abcdijkl', t2, t2)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)t2(a,d,j,k)*t2(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('adjk,bcil->abcdijkl', t2, t2)
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)t1(a,l)*t1(b,k)*t2(c,d,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('al,bk,cdij->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)t1(a,l)*t1(b,i)*t2(c,d,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('al,bi,cdjk->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,d)t1(a,l)*t1(d,k)*t2(b,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('al,dk,bcij->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->dbcaijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->dbcaikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,d)t1(a,l)*t1(d,i)*t2(b,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('al,di,bcjk->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->dbcaijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->dbcaijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,c)t1(a,k)*t1(b,l)*t2(c,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('ak,bl,cdij->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)t1(a,j)*t1(b,l)*t2(c,d,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('aj,bl,cdik->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)t1(a,j)*t1(b,i)*t2(c,d,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('aj,bi,cdkl->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)t1(a,j)*t1(d,l)*t2(b,c,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('aj,dl,bcik->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)t1(a,j)*t1(d,i)*t2(b,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('aj,di,bckl->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)t1(a,i)*t1(b,l)*t2(c,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ai,bl,cdjk->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)t1(b,l)*t1(c,k)*t2(a,d,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('bl,ck,adij->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)t1(b,l)*t1(c,i)*t2(a,d,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('bl,ci,adjk->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(c,d)t1(b,k)*t1(c,l)*t2(a,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bk,cl,adij->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)t1(b,j)*t1(c,l)*t2(a,d,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('bj,cl,adik->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)t1(b,j)*t1(c,i)*t2(a,d,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('bj,ci,adkl->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)t1(b,i)*t1(c,l)*t2(a,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bi,cl,adjk->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)t1(c,l)*t1(d,k)*t2(a,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('cl,dk,abij->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)t1(c,l)*t1(d,i)*t2(a,b,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('cl,di,abjk->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)t1(c,j)*t1(d,l)*t2(a,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('cj,dl,abik->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)t1(c,j)*t1(d,i)*t2(a,b,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('cj,di,abkl->abcdijkl', t1, t1, t2, optimize=['einsum_path', (0, 1, 2)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)t1(a,l)*t1(b,k)*t1(c,j)*t1(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('al,bk,cj,di->abcdijkl', t1, t1, t1, t1, optimize=['einsum_path', (0, 1, 2, 3)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,c)t1(a,l)*t1(b,i)*t1(c,k)*t1(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('al,bi,ck,dj->abcdijkl', t1, t1, t1, t1, optimize=['einsum_path', (0, 1, 2, 3)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)t1(a,k)*t1(b,l)*t1(c,j)*t1(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('ak,bl,cj,di->abcdijkl', t1, t1, t1, t1, optimize=['einsum_path', (0, 1, 2, 3)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)t1(a,j)*t1(b,l)*t1(c,k)*t1(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('aj,bl,ck,di->abcdijkl', t1, t1, t1, t1, optimize=['einsum_path', (0, 1, 2, 3)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)t1(a,j)*t1(b,i)*t1(c,l)*t1(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('aj,bi,cl,dk->abcdijkl', t1, t1, t1, t1, optimize=['einsum_path', (0, 1, 2, 3)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)t1(a,i)*t1(b,l)*t1(c,k)*t1(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('ai,bl,ck,dj->abcdijkl', t1, t1, t1, t1, optimize=['einsum_path', (0, 1, 2, 3)])
    t4 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 

    return t1, t2, t3, t4