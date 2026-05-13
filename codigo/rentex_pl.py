"""
PRO3341 - Projeto Rentex: Otimizacao do Mix de Producao
Modelo de Programacao Linear (PL) com analise de sensibilidade
Disciplina: PRO3341 - Modelagem e Otimizacao de Sistemas de Producao
Professora: Profa. Dra. Celma O. Ribeiro
Grupo: Guilherme de Jesus Lourencao (11260300)
       Emilly C C de Oliveira (15520464)
       Addison Fischer Borges (11375100)
"""
import pulp

# === DADOS ===
produtos = ['Sarja_Barcelos', 'Sarja_Joy', 'Forro_Padrao', 'Forro_Premium']
recursos = ['Teares', 'Fio_Algodao', 'Fio_Poliester', 'Elastano', 'MOD']

MC = {'Sarja_Barcelos': 700, 'Sarja_Joy': 1000,
      'Forro_Padrao': 220, 'Forro_Premium': 350}

b = {'Teares': 380, 'Fio_Algodao': 400, 'Fio_Poliester': 500,
     'Elastano': 12, 'MOD': 1200}

a = {
    'Teares':       {'Sarja_Barcelos':0.85,'Sarja_Joy':1.40,
                     'Forro_Padrao':0.50,'Forro_Premium':0.70},
    'Fio_Algodao':  {'Sarja_Barcelos':1.35,'Sarja_Joy':0.95,
                     'Forro_Padrao':0.00,'Forro_Premium':0.00},
    'Fio_Poliester':{'Sarja_Barcelos':0.00,'Sarja_Joy':0.35,
                     'Forro_Padrao':1.60,'Forro_Premium':1.55},
    'Elastano':     {'Sarja_Barcelos':0.00,'Sarja_Joy':0.07,
                     'Forro_Padrao':0.00,'Forro_Premium':0.08},
    'MOD':          {'Sarja_Barcelos':0.40,'Sarja_Joy':0.65,
                     'Forro_Padrao':0.25,'Forro_Premium':0.35},
}

d_min = {'Sarja_Barcelos':  0,'Sarja_Joy':  0,'Forro_Padrao': 60,'Forro_Premium': 20}
d_max = {'Sarja_Barcelos':200,'Sarja_Joy': 80,'Forro_Padrao':160,'Forro_Premium': 60}


def resolver(b_ov=None, MC_ov=None, inteiro=False):
    _b  = {**b,  **(b_ov  or {})}
    _MC = {**MC, **(MC_ov or {})}
    cat = 'Integer' if inteiro else 'Continuous'
    p = pulp.LpProblem("Rentex", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat=cat) for j in produtos}
    p += pulp.lpSum(_MC[j]*x[j] for j in produtos)
    for i in recursos:
        p += pulp.lpSum(a[i][j]*x[j] for j in produtos) <= _b[i], f"Cap_{i}"
    for j in produtos:
        p += x[j] >= d_min[j], f"Min_{j}"
        p += x[j] <= d_max[j], f"Max_{j}"
    p.solve(pulp.PULP_CBC_CMD(msg=0))
    return p, x


if __name__ == '__main__':
    prob, x = resolver()

    print("="*65)
    print("SOLUCAO OTIMA -- RENTEX MIX DE PRODUCAO")
    print("="*65)
    print(f"Status : {pulp.LpStatus[prob.status]}")
    print(f"Z*     : R$ {pulp.value(prob.objective):,.2f}/semana\n")

    print(f"{'Produto':<18} {'x*':>6} {'dmin':>5} {'dmax':>5}  {'MC unit':>8}  {'MC tot':>12}")
    print("-"*65)
    for j in produtos:
        xv = pulp.value(x[j])
        print(f"{j:<18} {xv:>6.1f} {d_min[j]:>5} {d_max[j]:>5}  {MC[j]:>8}  {MC[j]*xv:>12,.2f}")

    print(f"\n{'Recurso':<16}{'Uso':>8}{'Cap':>7}{'Folga':>8}{'Ocup%':>7}{'Pi(R$)':>10}")
    print("-"*56)
    for name, c in prob.constraints.items():
        if 'Cap' not in name: continue
        rec = name.replace('Cap_','')
        uso = sum(a[rec][j]*pulp.value(x[j]) for j in produtos)
        pi  = -c.pi if c.pi is not None else 0.0
        print(f"{rec:<16}{uso:>8.1f}{b[rec]:>7}{b[rec]-uso:>8.1f}{100*uso/b[rec]:>6.1f}%{pi:>10.2f}")

    # Ranging teares
    print("\n=== Ranging RHS -- Teares ===")
    z0 = pulp.value(prob.objective)
    print(f"{'Teares(h)':>10} {'Z*(R$)':>14} {'DeltaZ':>12}")
    for d in [-80,-60,-40,-20,0,20,40,60]:
        p2,_ = resolver(b_ov={'Teares': b['Teares']+d})
        zv = pulp.value(p2.objective)
        mark = " <- base" if d==0 else ""
        print(f"{b['Teares']+d:>10} {zv:>14,.2f} {zv-z0:>+12,.2f}{mark}")
