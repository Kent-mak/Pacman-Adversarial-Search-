from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        opt_action = self.minimax(0, 0, gameState)[1]
        return opt_action

    def minimax(self, agentIndex, cur_depth, gameState):
        if gameState.isWin() or gameState.isLose() or cur_depth > self.depth or(cur_depth == self.depth and agentIndex == 0):
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0: #MAX
            best_score = -float('inf')
            opt_action = None
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.getNextState(agentIndex, action)

                next_agent =  0 if agentIndex == gameState.getNumAgents() - 1 else agentIndex+1
                score = self.minimax(next_agent, cur_depth+1, nextState)[0]
                if score > best_score:
                    best_score, opt_action = score, action

            return best_score, opt_action
        else: #MIN
            best_score = float('inf')
            opt_action = None
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.getNextState(agentIndex, action)

                next_agent =  0 if agentIndex == gameState.getNumAgents() - 1 else agentIndex+1
                score = self.minimax(next_agent, cur_depth, nextState)[0]
                if score < best_score:
                    best_score, opt_action = score, action
                    
            return best_score, opt_action

        raise NotImplementedError("To be implemented")
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        opt_action = self.minimax_ABpruned(0, -float('inf'), float('inf'), 0, gameState)[1]
        return opt_action

    def minimax_ABpruned(self, agentIndex, alpha, beta, cur_depth, gameState):
        if gameState.isWin() or gameState.isLose() or cur_depth > self.depth or(cur_depth == self.depth and agentIndex == 0):
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0: #MAX
            best_score = -float('inf')
            opt_action = None
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.getNextState(agentIndex, action)
                next_agent =  0 if agentIndex == gameState.getNumAgents() - 1 else agentIndex+1
                score = self.minimax_ABpruned(next_agent,alpha, beta, cur_depth+1, nextState)[0]
                if score > best_score:
                    best_score, opt_action = score, action

                alpha = max(alpha, score)
                if alpha > beta:
                    break

            return best_score, opt_action
        
        else: #MIN
            best_score = float('inf')
            opt_action = None
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.getNextState(agentIndex, action)

                next_agent =  0 if agentIndex == gameState.getNumAgents() - 1 else agentIndex+1
                score = self.minimax_ABpruned(next_agent, alpha, beta, cur_depth, nextState)[0]
                if score < best_score:
                    best_score, opt_action = score, action

                beta = min(beta, score)
                if beta < alpha:
                    break
                
                    
            return best_score, opt_action
        raise NotImplementedError("To be implemented")
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        opt_action = self.expectimax(0, 0, gameState)[1]
        return opt_action
    
    def expectimax(self, agentIndex, cur_depth, gameState):
        if gameState.isWin() or gameState.isLose() or cur_depth > self.depth or(cur_depth == self.depth and agentIndex == 0):
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0: #MAX
            best_score = -float('inf')
            opt_action = None
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.getNextState(agentIndex, action)
                next_agent =  0 if agentIndex == gameState.getNumAgents() - 1 else agentIndex+1
                score = self.expectimax(next_agent, cur_depth+1, nextState)[0]
                if score > best_score:
                    best_score, opt_action = score, action

            return best_score, opt_action
        
        else: #Chance
            mean_score = 0
            count = 0
            for action in gameState.getLegalActions(agentIndex):
                nextState = gameState.getNextState(agentIndex, action)
                
                next_agent =  0 if agentIndex == gameState.getNumAgents() - 1 else agentIndex+1
                score = self.expectimax(next_agent,cur_depth, nextState)[0]
                mean_score += score
                count += 1
                    
            return mean_score /count , None
    # raise NotImplementedError("To be implemented")
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    num_food = currentGameState.getNumFood()

    if currentGameState.isLose(): 
        return -float("inf")
    elif currentGameState.isWin():
        if num_food == 0:
            return float("inf")
        else:
            return 1e6
    
    
    baseScore = scoreEvaluationFunction(currentGameState)

    pacState = currentGameState.getPacmanState()
    pacPos = pacState.getPosition()

    ghostsState = currentGameState.getGhostStates()

    foodlist = currentGameState.getFood().asList()
    
    minDistToFood = min(map(lambda pos: util.manhattanDistance(pacPos, pos), foodlist))
    num_capsules = len(currentGameState.getCapsules())

    

    active_ghosts = []
    scared_ghosts = []
    for ghost in ghostsState:
        if ghost.scaredTimer > 0:
            scared_ghosts.append(ghost)
        else:
            active_ghosts.append(ghost)
    
    minDistToActiveGhost = float('inf')
    if len(active_ghosts) > 0:
        minDistToActiveGhost = min(map(lambda ghost: util.manhattanDistance(pacPos, ghost.getPosition()), active_ghosts))
    
    minDistToScaredGhost = 0
    if len(scared_ghosts) > 0:
        minDistToScaredGhost = min(map(lambda ghost: util.manhattanDistance(pacPos, ghost.getPosition()), scared_ghosts))

    score = -2*minDistToFood - 10*minDistToScaredGhost - 2/minDistToActiveGhost - 50*num_food - 200*num_capsules
    
    return score
    raise NotImplementedError("To be implemented")
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
