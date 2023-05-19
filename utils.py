import numpy as np

def first_visit_MC(env, stateNumber, numberOfEpisodes, discountRate):
    V = np.zeros(stateNumber)
    returnState = np.zeros(stateNumber)
    visitedStatesNum = np.zeros(stateNumber)

    # episode simulation
    for episode in range(numberOfEpisodes):
        visitedStates = []
        rewardState = []
        currentState = env.reset()
        visitedStates.append(currentState)

        print("Simulating episode {}".format(episode))

        while True:
            randomAction = env.action_space.sample()
            currentState, currentReward, terminalState = env.step(randomAction)
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