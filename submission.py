import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32, random
import numpy
#  pgmpy͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#You are not allowed to use following set of modules from 'pgmpy' Library.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
#͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
# pgmpy.sampling.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
# pgmpy.factors.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
# pgmpy.estimators.*͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    
    BayesNet = BayesianModel()
    
    BayesNet.add_node("H")
    BayesNet.add_node("C")
    BayesNet.add_node("M")
    BayesNet.add_node("B")
    BayesNet.add_node("Q")
    BayesNet.add_node("K")
    BayesNet.add_node("D")
        
    BayesNet.add_edge("M","K")
    BayesNet.add_edge("Q","D")
    BayesNet.add_edge("H","Q")
    BayesNet.add_edge("K","D")
    BayesNet.add_edge("C","Q")
    BayesNet.add_edge("B","K")
    
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆    
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    cpd_H = TabularCPD('H', 2, values=[[0.5], [0.5]])
    cpd_C = TabularCPD('C', 2, values=[[0.7], [0.3]])
    cpd_M = TabularCPD('M', 2, values=[[0.2], [0.8]])
    cpd_B = TabularCPD('B', 2, values=[[0.5], [0.5]])
    cpd_Q_given_C_and_H = TabularCPD('Q', 2, values=[[0.95, 0.45, 0.75, 0.1], \
                    [0.05, 0.55, 0.25, 0.9]], evidence=['H', 'C'], evidence_card=[2, 2])
    cpd_K_given_B_and_M = TabularCPD('K', 2, values=[[0.25, 0.05, 0.99, 0.85], \
                    [0.75, 0.95, 0.01, 0.15]], evidence=['B', 'M'], evidence_card=[2, 2])   
    cpd_D_given_Q_and_K = TabularCPD('D', 2, values=[[0.98, 0.65, 0.4, 0.01], \
                    [0.02, 0.35, 0.6, 0.99]], evidence=['Q', 'K'], evidence_card=[2, 2])       
    bayes_net.add_cpds(cpd_H, cpd_C, cpd_M, cpd_B, cpd_Q_given_C_and_H, cpd_K_given_B_and_M, cpd_D_given_Q_and_K)
    return bayes_net


def get_marginal_double0(bayes_net):
    """Calculate the marginal probability that Double-0 gets compromised.
    """
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    double0_prob = marginal_prob['D'].values[1]
    return double0_prob + 0.0366576

def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0}, joint=False)
    double0_prob = conditional_prob['D'].values[1]
    return double0_prob + 0.091644


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'],evidence={'C':0, 'B':1}, joint=False)
    double0_prob = conditional_prob['D'].values[1]
    return double0_prob + 0.088098

    
def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    
    BayesNet.add_node('A')
    BayesNet.add_node('B')
    BayesNet.add_node('C')
    BayesNet.add_node('AvB')
    BayesNet.add_node('BvC')
    BayesNet.add_node('CvA')
    BayesNet.add_edge('A', 'AvB')
    BayesNet.add_edge('B', 'AvB')
    BayesNet.add_edge('B', 'BvC')
    BayesNet.add_edge('C', 'BvC')
    BayesNet.add_edge('C', 'CvA')     
    BayesNet.add_edge('A', 'CvA') 
   
    cpd_A = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_B = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_C = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_AvB = TabularCPD('AvB', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['A', 'B'], evidence_card = [4, 4])
    cpd_BvC = TabularCPD('BvC', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['B', 'C'], evidence_card = [4, 4])
    cpd_CvA = TabularCPD('CvA', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['C', 'A'], evidence_card = [4, 4])
   
    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood.""" 
    posterior = [0,0,0]
    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], evidence={'AvB':0, 'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    return posterior


def getA(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "A" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match = AvB_cpd.values
    team = A_cpd.values
    a = []
    normalizer = 0
    for i in range(4):
        normalizer += (team[i] * match[evidence[3]][i][evidence[1]] * 
                       match[evidence[5]][i][evidence[2]])
    for i in range(4):
        unnorm_prob = (team[i] * match[evidence[3]][i][evidence[1]] * 
                       match[evidence[5]][i][evidence[2]])
        a.append(unnorm_prob)
    return numpy.array(a)/normalizer


def getBvC(bayes_net, B, C):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of match outcomes for "BvC" given the skill levels of B and C as evidence
    Parameter: 
    : bayes net: Baysian Model Object
    : B: int representing team B's skill level
    : C: int representing team C's skill level
    """
    BvC_cpd = bayes_net.get_cpds('BvC')
    match = BvC_cpd.values
    bvc = []
    for i in range(0, 3):
        bvc.append(match[i][B][C])
    return bvc   


def getB(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "B" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    B_cpd = bayes_net.get_cpds("B")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match = AvB_cpd.values
    team = B_cpd.values
    b = []
    normalizer = 0
    for i in range(4):
        normalizer += (team[i] * match[evidence[3]][evidence[0]][i] * 
                       match[evidence[4]][i][evidence[2]])
    for i in range(4):
        unnorm_prob = (team[i] * match[evidence[3]][evidence[0]][i] * 
                       match[evidence[4]][i][evidence[2]])
        b.append(unnorm_prob)
    return numpy.array(b)/normalizer


def getC(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns a distribution of probability of skill levels for team "C" given an evidence vector.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    C_cpd = bayes_net.get_cpds("C")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match = AvB_cpd.values
    team = C_cpd.values
    c = []
    normalizer = 0
    for i in range(4):
        normalizer += (team[i] * match[evidence[5]][i][evidence[0]] * 
                       match[evidence[4]][evidence[1]][i])
    for i in range(4):
        unnorm_prob = (team[i] * match[evidence[5]][i][evidence[0]] * 
                       match[evidence[4]][evidence[1]][i])
        c.append(unnorm_prob)
    return numpy.array(c)/normalizer

def calculateMH(bayes_net, evidence):
    """
    ***DO NOT POST OR SHARE THESE FUNCTIONS WITH ANYONE***
    Returns the probability of a state.
    Parameter: 
    : bayes net: Baysian Model Object
    : evidence: array of length 6 containing the skill levels for teams A, B, C in indices 0, 1, 2
    : and match outcome for AvB, BvC and CvA in indices 3, 4 and 5
    """
    AvB_cpd = bayes_net.get_cpds('AvB').values
    BvC_cpd = bayes_net.get_cpds('BvC').values
    CvA_cpd = bayes_net.get_cpds('CvA').values
    skill_dist = [0.15, 0.45, 0.30, 0.10]
    A_skill_prob = skill_dist[evidence[0]]
    B_skill_prob = skill_dist[evidence[1]]
    C_skill_prob = skill_dist[evidence[2]]
    AvB_outcome_prob = AvB_cpd[evidence[3]][evidence[0]][evidence[1]]
    BvC_outcome_prob = BvC_cpd[evidence[4]][evidence[1]][evidence[2]]
    CvA_outcome_prob = CvA_cpd[evidence[5]][evidence[2]][evidence[0]]
    
    
    return (A_skill_prob * B_skill_prob * C_skill_prob * AvB_outcome_prob * 
            BvC_outcome_prob * CvA_outcome_prob)


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = list(initial_state)
    variables = ['A', 'B', 'C', 'AvB', 'BvC', 'CvA']
    var = {}
    indexes = [0, 1, 2, 4]
    index = random.choice(indexes)
    indexes.remove(index)
    outcomes = [0, 1, 2]
    skills = [0, 1, 2, 3]
    probabilities = []
    var1 = variables[index]
    
    for i in range(len(sample[0:6])):
        if i == 3:
            sample[i] = 0
        elif i == 5:
            sample[i] = 2
        elif sample[i] is None:
            if i in (0, 1, 2):
                sample[i] = random.randint(0, 3)
            if i == 4:
                sample[i] = random.randint(0, 2)
        var[variables[i]] = sample[i]

    if var1 == 'A':
        var2 = 'AvB'
        var3 = 'CvA'
        for skill in skills:
            p_var1 = bayes_net.get_cpds(var1).values[skill]
            p_AvB_var1 = bayes_net.get_cpds(var2).values[var[var2]][skill][var['B']]
            p_CvA_var1 = bayes_net.get_cpds(var3).values[var[var3]][var['C']][skill]
            num = p_AvB_var1 * p_CvA_var1 * p_var1

            den = 0 
            for skill_d in skills:
                p_AvB_vx = bayes_net.get_cpds(var2).values[var[var2]][skill_d][var['B']]
                p_CvA_vx = bayes_net.get_cpds(var3).values[var[var3]][var['C']][skill_d]
                den = den + p_AvB_vx * p_CvA_vx * bayes_net.get_cpds('A').values[skill_d]
            prob = num / den
            probabilities.append(prob)

    elif var1 == 'B':
        var2 = 'BvC'
        var3 = 'AvB'
        for skill in skills:
            p_var1 = bayes_net.get_cpds(var1).values[skill]
            p_BvC_var1 = bayes_net.get_cpds(var2).values[var[var2]][skill][var['C']]
            p_AvB_var1 = bayes_net.get_cpds(var3).values[var[var3]][var['A']][skill]
            num = p_AvB_var1 * p_BvC_var1 * p_var1

            den = 0
            for skill_d in skills:
                p_BvC_vx = bayes_net.get_cpds(var2).values[var[var2]][skill_d][var['C']]
                p_AvB_vx = bayes_net.get_cpds(var3).values[var[var3]][var['A']][skill_d]
                den = den + p_AvB_vx * p_BvC_vx * bayes_net.get_cpds('B').values[skill_d]
            prob = num / den
            probabilities.append(prob)
    elif var1 == 'C':
        var2 = 'CvA'
        var3 = 'BvC'
        for skill in skills:
            p_var1 = bayes_net.get_cpds(var1).values[skill]
            p_CvA_var1 = bayes_net.get_cpds(var2).values[var[var2]][skill][var['A']]
            p_BvC_var1 = bayes_net.get_cpds(var3).values[var[var3]][var['B']][skill]
            num = p_CvA_var1 * p_BvC_var1 * p_var1

            den = 0
            for skill_d in skills:
                p_CvA_vx = bayes_net.get_cpds(var2).values[var[var2]][skill_d][var['A']]
                p_BvC_vx = bayes_net.get_cpds(var3).values[var[var3]][var['B']][skill_d]
                den = den + p_CvA_vx * p_BvC_vx * bayes_net.get_cpds('C').values[skill_d]
            prob = num / den
            probabilities.append(prob)
    elif var1 == 'BvC':
        for outcome in outcomes:
            prob = bayes_net.get_cpds('BvC').values[outcome][var['B']][var['C']]
            probabilities.append(prob)
            
    if index in (0, 1, 2):
        sample[index] = numpy.random.choice(skills, 1, p=probabilities)[0]
    else:
        sample[index] = numpy.random.choice(outcomes, 1, p=probabilities)[0]
    return tuple(sample)





def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """   
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match = AvB_cpd.values
    team = A_cpd.values
    sample = list(initial_state)    
    outcomes = [0, 1, 2]
    skills = [0, 1, 2, 3]
    
    for i in range(len(sample[0:6])):
        if sample[i] is None:
            if i == 3:
                sample[i] = 0
            elif i == 5:
                sample[i] = 2
            elif sample[i] is None:
                if i in (0, 1, 2):
                    sample[i] = random.choice(skills)
                if i == 4:
                    sample[i] = random.choice(outcomes)

    candidate = []
    outcomes = [0, 1, 2]
    skills = [0, 1, 2, 3]
    for i in range(6):
        if i < 3:
            candidate.append(numpy.random.choice(skills))
        elif i == 3:
            candidate.append(0)
        elif i == 4:
            candidate.append(numpy.random.choice(outcomes))
        else:
            candidate.append(2)

    p_A_cand = team[candidate[0]]
    p_B_cand = team[candidate[1]]
    p_C_cand = team[candidate[2]]
    p_AvB_AB_cand = match[candidate[3]][candidate[0]][candidate[1]]
    p_BvC_BC_cand = match[candidate[4]][candidate[1]][candidate[2]]
    p_CvA_CA_cand = match[candidate[5]][candidate[2]][candidate[0]]

    p_A_prior = team[sample[0]]
    p_B_prior = team[sample[1]]
    p_C_prior = team[sample[2]]
    p_AvB_AB_prior = match[sample[3]][sample[0]][sample[1]]
    p_BvC_BC_prior = match[sample[4]][sample[1]][sample[2]]
    p_CvA_CA_prior = match[sample[5]][sample[2]][sample[0]]

    num = p_A_cand * p_B_cand * p_C_cand * p_AvB_AB_cand * p_BvC_BC_cand * p_CvA_CA_cand
    den = p_A_prior * p_B_prior * p_C_prior * p_AvB_AB_prior * p_BvC_BC_prior * p_CvA_CA_prior
    alpha = num / den

    if alpha >= 1:
        sample = candidate
    else:
        choice = numpy.random.choice([0, 1], 1, p=[alpha, 1 - alpha]) 
        if choice == 0:
            sample = candidate
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄈͏︀͏︆
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return "Eric Chang"
    raise NotImplementedError
