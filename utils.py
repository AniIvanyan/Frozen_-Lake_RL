import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  

# Learns a value function for an environment using first-visit Monte Carlo control.
def first_visit_MC(env, stateNumber, numberOfEpisodes, discountRate):
    V = np.zeros(stateNumber)
    returnState = np.zeros(stateNumber)
    visitedStatesNum = np.zeros(stateNumber)

    # episode simulation
    for episode in range(numberOfEpisodes):
        visitedStates = []
        rewardState = []
        (currentState,_) = env.reset()
        visitedStates.append(currentState)

        print("Simulating episode {}".format(episode))

        while True:
            randomAction = env.action_space.sample()
            (currentState, currentReward, terminalState,_,_)= env.step(randomAction)
            rewardState.append(currentReward)
            if not terminalState:
                visitedStates.append(currentState)
            else:
                break

        numVisitedState = len(visitedStates)
        G = 0
        for eachstep in range(numVisitedState - 1, -1, -1):
            episodeState = visitedStates[eachstep]
            episodeReturn = rewardState[eachstep]
            G = discountRate * G + episodeReturn

            if episodeState not in visitedStates[0:eachstep]:
                visitedStatesNum[episodeState] += 1
                returnState[episodeState] += G

    for i in range(stateNumber):
        if visitedStatesNum[i] != 0:
            V[i] = returnState[i] / visitedStatesNum[i]
    return V

# Iteratively updates a value function for an environment using a given policy.
def iterative_policy_evaluation(env, valueFunction, policy, discountRate, maxNumberOfIterations, convergenceTolerance):
    for i in range(maxNumberOfIterations):
        valueFunctionNextIteration = np.zeros(env.observation_space.n)
        for state in env.P:
            outerSum = 0
            for action in env.P[state]:
                innerSum = 0
                for probability, nextState, reward, _ in env.P[state][action]:
                    innerSum += probability * (reward + discountRate * valueFunction[nextState])
                outerSum += policy[state, action] * innerSum
            valueFunctionNextIteration[state] = outerSum
        if np.max(np.abs(valueFunctionNextIteration - valueFunction)) < convergenceTolerance:
            valueFunction = valueFunctionNextIteration
            print('Iterative policy evaluation algorithm converged!')
            break
        valueFunction = valueFunctionNextIteration
    return valueFunction

# Visualizes a value function as a heatmap.
def visualize_valueFuntion(valueFunction,shape,fileNameToSave):
    ax = sns.heatmap(valueFunction.reshape(shape,shape),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig(fileNameToSave,dpi=600)
    plt.show()