#sage script

q = 2^64
p = 2^4

# GenPBS param
N = 2^11 #Poly length
k = 1 #GLWE comps
n = 742 #LWE length
Var_RLWE = (2.94036e-16)^2 #BSK error variance
Var_LWE = (7.06984e-6)^2 #KSK error variance
Var_BSK = Var_RLWE * q^2
Var_KSK = Var_LWE * q^2
B_pbs = 2^23 #Decomp. Base
l_pbs = 1 #Decomp. level
B_ksk = 2^3
l_ksk = 5

B2_12_pbs = (B_pbs^2 + 2)/12
Bp_2l_pbs = B_pbs^(2*l_pbs)

B2_12_ksk = (B_ksk^2 + 2)/12
Bp_2l_ksk = B_ksk^(2*l_ksk)

Var_PBS = 0
Var_PBS += n*l_pbs*(k+1)*N*B2_12_pbs*Var_BSK
Var_PBS += n*(q^2-Bp_2l_pbs)/(24*Bp_2l_pbs) * (1+k*N/2)
Var_PBS += (n*k*N)/32
Var_PBS += (n/16)*(1-k*N/2)^2

# KS param
Var_KS = 0
Var_KS += k*N * (q^2/(12*Bp_2l_ksk)-1/12) * (1/2)
Var_KS += k*N/16
Var_KS += k*N*l_ksk*Var_KSK*B2_12_ksk


# GLWEtoGLWE KS param
B_subs = 2^9
l_subs = 5
Var_subs = Var_RLWE * q^2

V_sk = 0
V_sk += k*N/2 * (q^2 / (12 * B_subs^(2*l_subs)) - 1 / 12)
V_sk += k*N/16
V_sk += k*N*l_subs*Var_subs*(B_subs^2 + 2)/12

Var_subs_tot = N^2 * Var_subs + (N^2 - 1)/3 * V_sk

# GGSW(-S) param
B_ggsw_sk = 2^9
l_ggsw_sk = 5
Var_ggsw_sk = Var_RLWE * q^2

B_rk = 2^7
l_rk = 3

Var_dbr = 0
Var_dbr += l_ggsw_sk*(k+1)*N*(B_ggsw_sk^2 + 2)/12*Var_ggsw_sk
Var_dbr += Var_subs_tot
Var_dbr += (q^2 - B_rk^(2*l_ggsw_sk))/(12*B_ggsw_sk^(2*l_ggsw_sk))*(1 + k*N/2)
Var_dbr += k*N/8
Var_dbr += ((1-k*N/2)^2)/4

# Final
V_DBR_4 = 0
V_DBR_4 += 4^2 * (l_rk*(k+1)*N*(B_rk^2+2)/12*Var_dbr)
V_DBR_4 += 4 * (q^2 - B_pbs^(2*l_pbs))/(12 * B_pbs^(2*l_pbs))*(1+k*N/2)
V_DBR_4 += 4 * k*N/8
V_DBR_4 += 4 * ((1-k*N/2)^2)/4

V_DBR_3 = 0
V_DBR_3 += 3^2 * (l_rk*(k+1)*N*(B_rk^2+2)/12*Var_dbr)
V_DBR_3 += 3 * (q^2 - B_pbs^(2*l_pbs))/(12 * B_pbs^(2*l_pbs))*(1+k*N/2)
V_DBR_3 += 3 * k*N/8
V_DBR_3 += 3 * ((1-k*N/2)^2)/4

Var_crfsum = 2*31^2*Var_PBS + 30*(V_DBR_4 + V_DBR_3) + 3 * Var_PBS + Var_KSK
print("V_crfsum: ", (log(Var_crfsum)/log(2)).n())
print("- V_sk: ", (log(V_sk)/log(2)).n())
print("- Var_subs_tot: ", (log(Var_subs_tot)/log(2)).n())
print("- Var_PBS: ", (log(Var_PBS)/log(2)).n())
print("- Var_BSK: ", (log(Var_BSK)/log(2)).n())
print("- Var_KSK: ", (log(Var_KSK)/log(2)).n())
print("- Var_dbr: ", (log(Var_dbr)/log(2)).n())
print("- V_DBR_3: ", (log(V_DBR_3)/log(2)).n())
print("- V_DBR_4: ", (log(V_DBR_4)/log(2)).n())


theta = 1
w = 2*N/(2**theta)
q_prime = q
delta_in = 2^60
Gamma = (delta_in / 2) * (Var_crfsum + q_prime^2/(12*w^2) - 1/12 + n*q_prime^2/(24*w^2) + n/48)^(-1/2)

print("Gamma:", Gamma)
print()

sq2 = 2^(1/2)

res = erf(Gamma/sq2)
fail = 1 - res
print("Failure Prob (Single Round): 2^", log(fail, 2).n(100))
print("Failure Prob (FRAST)       : 2^", log(fail * 40, 2).n(100))
