"""
PRO3341 -- Projeto Rentex: Otimizacao do Mix de Producao
Modelo de Programacao Linear (PL) com analise de sensibilidade
Disciplina: PRO3341 - Modelagem e Otimizacao de Sistemas de Producao
Professora: Profa. Dra. Celma O. Ribeiro
Grupo: Guilherme de Jesus Lourencao (11260300)
       Emilly C C de Oliveira (15520464)
       Addison Fischer Borges (11375100)
"""

import pulp

# ===========================================================================
# DADOS DO PROBLEMA
# ===========================================================================
produtos = ['Jersey', 'Rib', 'Popeline', 'Oxford', 'Denim', 'Tecnico']
recursos = ['Teares', 'Tinturaria', 'Acabamento', 'Fio', 'MOD']
MC = {'Jersey': 97, 'Rib': 119, 'Popeline': 127,
      'Oxford': 160, 'Denim': 180, 'Tecnico': 223}
b  = {'Teares': 1180, 'Tinturaria': 775, 'Acabamento': 570,
      'Fio': 16000, 'MOD': 890}
a  = {
    'Teares':     {'Jersey':1.8,'Rib':2.3,'Popeline':2.9,'Oxford':3.4,'Denim':4.1,'Tecnico':3.6},
    'Tinturaria': {'Jersey':1.4,'Rib':1.6,'Popeline':1.9,'Oxford':2.4,'Denim':2.9,'Tecnico':1.1},
    'Acabamento': {'Jersey':0.7,'Rib':0.9,'Popeline':1.1,'Oxford':1.3,'Denim':1.7,'Tecnico':2.6},
    'Fio':        {'Jersey': 17,'Rib': 21,'Popeline': 24,'Oxford': 29,'Denim': 36,'Tecnico': 27},
    'MOD':        {'Jersey':1.1,'Rib':1.4,'Popeline':1.7,'Oxford':1.9,'Denim':2.4,'Tecnico':2.9},
}
d = {'Jersey':195,'Rib':148,'Popeline':118,'Oxford':98,'Denim':78,'Tecnico':58}

def resolver_pl(b_override=None, MC_override=None, inteiro=False):
    _b  = {**b,  **(b_override  or {})}
    _MC = {**MC, **(MC_override or {})}
    cat = 'Integer' if inteiro else 'Continuous'
    prob = pulp.LpProblem("Rentex_Mix", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat=cat) for j in produtos}
    prob += pulp.lpSum(_MC[j] * x[j] for j in produtos)
    for i in recursos:
        prob += (pulp.lpSum(a[i][j] * x[j] for j in produtos) <= _b[i], f"Cap_{i}")
    for j in produtos:
        prob += x[j] <= d[j], f"Dem_{j}"
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob, x

if __name__ == '__main__':
    prob, x = resolver_pl()
    print(f"Z* = R$ {pulp.value(prob.objective):,.2f}")
    for j in produtos:
        print(f"  {j}: {pulp.value(x[j]):.2f} lotes")
    print()
    for name, constr in prob.constraints.items():
        if 'Cap' in name:
            rec = name.replace('Cap_','')
            uso = sum(a[rec][j]*pulp.value(x[j]) for j in produtos)
            pi  = -constr.pi if constr.pi is not None else 0.0
            print(f"  {rec}: uso={uso:.1f}  pi=R${pi:.4f}")
