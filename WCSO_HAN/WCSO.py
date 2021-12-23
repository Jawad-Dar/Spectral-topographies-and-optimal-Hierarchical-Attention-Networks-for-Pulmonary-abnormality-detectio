import random, numpy as np

def algm():
    lb, ub, C, dmax = 0, 1, 2, 0.5  # lower, upper bound
    n = 5
    Npop, Nvar = (2 * n) + 1, 5  # row(raindrop-n river, n stream, 1 sea), column(design variables)
    Nsr = n + 1  # best individuals (No. of rivers+1 sea) -- user parameter
    Nrd = Npop - Nsr  # eqn. (5) Nraindrops -- other than best
    g, max_itr = 0, 10  # initial, max iteration

    def fitness(soln):
        Fit = []
        for i in range(len(soln)):
            summ, hr = 0, random.random()
            for j in range(len(soln[i])):
                summ += soln[i][j]
            Fit.append(1/(summ*hr))
        return Fit


    # split raindrop in sea, river, stream
    def split_raindrop(r_drop, sr, l, u):
        sea, river, stream = [], [], []
        for i in range(len(r_drop)):  # all raindrop
            if i < sr:
                if i == 0:
                    sea.append(r_drop[i])  # first best raindrop in sea
                else:
                    river.append(r_drop[i])  # other best raindrop(in Nsr) in river
            else:
                stream.append(r_drop[i])  # other than best in stream
        return bound(sea, l, u), bound(river, l, u), bound(stream, l, u)


    # sort Population by fitness
    def sort_by_cost(ft, raindrop):
        sorted_fit, sorted_raindrop = [], []
        for i in range(len(ft)):
            bst_index = np.argmin(ft)  # min. fit value index
            sorted_fit.append(ft[bst_index])  # sort fit
            sorted_raindrop.append(raindrop[bst_index])  # sort raindrop
            ft[bst_index] = float('inf')  # set by max. value to get next best
        return sorted_fit, sorted_raindrop


    # Assigning raindrop to river & sea depends on intensity of flow
    def intensity_of_flow(cost, sr, rd):
        Sn, Ecost = [], sum(cost)
        for i in range(int(sr)):
            Sn.append(round(np.abs(cost[i] / Ecost) * rd))  # eqn. (6)
        return Sn


    # update the stream
    def update_stream(stm, rvr, C):
        stm_upd = []  # updated stream
        for i in range(len(stm)):
            tem, rand = [], random.random( )
            for j in range(len(stm[i])):
                tem.append(stm[i][j] + (rand * C * (rvr[i][j] - stm[i][j])))  # eqn. (8)
            stm_upd.append(tem)
        return stm_upd


    # update the river
    def update_river(rvr, sea, C):
        def mean_pos(P):  # mean position of Particles
            temp = []
            mean_pos = []
            for i in range(len(P)):
                for j in range(len(P)):
                    temp.append(P[j][i])
                mean_pos.append(np.mean(temp))
            return mean_pos
        m_P = mean_pos(rvr)
        rvr_upd = []  # updated stream
        phi = np.random.uniform(0, 1)  # parameter which controls the influence of Xk
        R1,R2,R3 = random.random(),random.random(),random.random()
        for i in range(len(rvr)):
            V = np.zeros((Npop, Nvar))  # initial velocity
            tem, rand = [], random.random( )
            for j in range(len(rvr[i])):
                ##############Proposed updated equation###############
                new = (((1-R2-phi-R3)/(rand*C-R2-phi*R3))*rand*C*sea[0][j]-((R1*V[i][j]+R2*rvr[i][j]+phi*R3*m_P[i])/(1-R2-phi*R3))*(1-rand*C)).tolist()
                tem.append(new)
            rvr_upd.append(tem)
        return rvr_upd


    # random generation of data
    def generate(n, m, l, u):
        data = []
        for i in range(n):
            tem = []
            for j in range(m):
                tem.append(random.uniform(l, u))
            data.append(tem)
        return data


    # Update Pop (Sea+River+Stream)
    def update_pop(S, R, St):
        pop_up = []
        for i in range(len(S)): pop_up.append(S[i])
        for i in range(len(R)): pop_up.append(R[i])
        for i in range(len(St)): pop_up.append(St[i])
        return pop_up


    # bound within limit
    def bound(data, l, u):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] < l or data[i][j] > u:  # out of limit
                    data[i][j] = l + (random.random( ) * (u - l))
        return data

    # Raining Process
    def raining_process(sea, stream, l, u):
        bound(stream, l, u)
        new_stream, mu = [], 0.1
        for i in range(len(stream)):
            tem = []
            for j in range(len(stream[i])):
                randn = random.randint(0, len(stream[i]) - 1)  # rand[1,Nvar]
                tem.append(sea[0][j] + (np.sqrt(mu) * stream[i][randn]))     # eqn. (12)
            new_stream.append(tem)
        return new_stream


    Pop = generate(Npop, Nvar, lb, ub)  # initial population(raindrops)


    # Loop begins
    overall_fit, overall_best = [], []
    while g < max_itr:
        Fit = fitness(Pop)  # cost (fitness function)  -- eq(3)
        fit_tem = Fit.copy()
        fit, Pop = sort_by_cost(fit_tem, Pop)  # sort raindrop in ascending order
        overall_fit.append(fit[0])
        overall_best.append(Pop[0])
        Sea, River, Stream = split_raindrop(Pop, Nsr, lb, ub)  # split raindrop in sea, river & stream
        Nsn = intensity_of_flow(fit, Nsr, Nrd)  # flow intensity

        Stream = update_stream(Stream, River, C)  # stream update
        River = update_river(River, Sea, C)  # river update
        Pop = update_pop(Sea, River, Stream)

        if np.abs(np.mean(Sea) - (np.mean(River))) < dmax:
            Stream = raining_process(Sea, Stream, lb, ub)
        dmax = dmax - (dmax / max_itr)

        g += 1
    bst = np.argmin(overall_fit)
    return np.max(overall_best[bst])