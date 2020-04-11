import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import sys
import cv2
import math 

plt.ion()
np.set_printoptions(precision = 3,suppress = True)


def get_alpha(theta):
    print("Theta :",theta)
    while theta>360:
        theta = theta-360
    alpha = int(theta/30)
    if alpha<0:
        alpha +=12

    if alpha>=12:
        alpha -=12
    return alpha    

def plot_curve(Xi,Yi,Thetai,UL,UR,c = "blue"):
    t = 0
    r = 3.8
    L = 35.4
    dt = 0.1
    Xn=Xi
    Yn=Yi
    Thetan = 3.14 * Thetai / 180

    # Xi, Yi,Thetai: Input point's coordinates
    # Xs, Ys: Start point coordinates for plot function
    # Xn, Yn, Thetan: End point coordintes

    while t<1:
        t = t + dt
        Xs = Xn
        Ys = Yn
        Xn += 0.5 * r * (UL + UR) * math.cos(Thetan) * dt
        Yn += 0.5 * r * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        plt.plot([Xs, Xn], [Ys, Yn], color=c)
    Thetan = 180 * (Thetan) / 3.14
    # Thetan = get_alpha(Thetan)
    return Xn, Yn, Thetan

class Map:
    def __init__(self,start,goal):
        """
        Constructs a new instance.
    
        :param      start:      The start
        :type       start:      Start Node coordinates
        :param      goal:       The goal
        :type       goal:       Goal Node coordinates
        :param      clearence:  The clearence
        :type       clearence:  Int
        :param      radius:     The radius
        :type       radius:     Int
        :param      step_size:  The step size
        :type       step_size:  Int
        """
        self.visited_map = np.zeros((2040,2040,12),np.uint8)
        self.start = start
        self.goal = goal
        self.queue=[]
        self.visited = []
        self.shortest_path = [] 
        self.radius= 36
        self.clearence= 5
        r = self.radius
        c = self.clearence
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        

        plt.scatter(start[0],start[1],s = 10,c = 'r')
        plt.scatter(goal[0],goal[1],s = 10,c = 'r')
        # Circles
        plt.gca().add_patch(plt.Circle((510,510), radius=100, fc='y'))
        plt.gca().add_patch(plt.Circle((710,810), radius=100, fc='y'))
        plt.gca().add_patch(plt.Circle((310,210), radius=100, fc='y'))
        plt.gca().add_patch(plt.Circle((710,210), radius=100, fc='y'))

        # Boundary
        plt.gca().add_patch(plt.Rectangle((0,0),10,1020, color='black'))
        plt.gca().add_patch(plt.Rectangle((0,0),1020,10, color='black'))
        plt.gca().add_patch(plt.Rectangle((0,1010),1020,10,color='black'))
        plt.gca().add_patch(plt.Rectangle((1010,0),10,1020,color='black'))

        # Squares
        plt.gca().add_patch(plt.Rectangle((235,735),150,150))
        plt.gca().add_patch(plt.Rectangle((35,460),150,150))
        plt.gca().add_patch(plt.Rectangle((835,460),150,150))
        plt.axis('scaled')


    def isObstacle(self,x,y):
        """
        Determines if obstacle using half-plane equations.
    
        :param      j:    x-coordinate
        :type       j:    flloat
        :param      i:    y-coordinate
        :type       i:    float
    
        :returns:   True if obstacle, False otherwise.
        :rtype:     boolean
        """
        r=self.radius
        c=self.clearence

        obstacle = False
      
        if (x<=(10+r+c)) or (x>=1020-(10+(r+c))) or (y<=(10+r+c)) or (y>=1020-(10+(r+c))):
            print("Boundary Condition")
            obstacle = True

       # Circle
        if ((x-510)**2 + (y-510)**2 <= ((100+r+c)**2)):
            print("Circle Condition 1")
            obstacle = True
        
        # Circle
        if ((x-710)**2 + (y-810)**2 <= ((100+r+c)**2)):
            print("Circle Condition 2")
            obstacle = True
        
        # Circle
        if ((x-310)**2 + (y-210)**2 <= ((100+r+c)**2)):
            print("Circle Condition 3")
            obstacle = True

        # Circle
        if ((x-710)**2 + (y-210)**2 <= ((100+r+c)**2)):
            print("Circle Condition 4")
            obstacle = True

        # Rectangle
        if (((x-(35-(r+c)))>=0) and ((x-(185+(r+c)))<=0) and ((y-(460-(r+c)))>=0) and ((y-(660+(r+c)))<=0)):
            print("Rectangle Condition 1")
            obstacle = True

        # Rectangle
        if (((x-(235-(r+c)))>=0) and ((x-(385+(r+c)))<=0) and ((y-(735-(r+c)))>=0) and ((y-(885+(r+c)))<=0)):
            print("Rectangle Condition 2")
            obstacle = True

        # Rectangle
        if (((x-(835-(r+c)))>=0) and ((x-(985+(r+c)))<=0) and ((y-(460-(r+c)))>=0) and ((y-(660+(r+c)))<=0)):
            print("Rectangle Condition 3")
            obstacle = True


        return obstacle
        
    def cost(self,node,step_cost):
        """
        Returns 
    
        :param      node:       The node at which the cost is to be obtained 
        :type       node:       List
        :param      step_cost:  The step cost to reach that node from start node
        :type       step_cost:  int
    
        :returns:   Heuristic cost for A*
        :rtype:     float
        """
        return(step_cost + np.linalg.norm(np.array(node[0:2])-np.array(self.goal[0:2]),2))



    def actionsAvailable(self,x,y,theta,step_cost):
        """
        Returns 5 actions available at the given nodes 
    
        :param      x:          The current node x
        :type       x:          float
        :param      y:          The current node y
        :type       y:          float
        :param      theta:      The theta
        :type       theta:      Int in range (0,12)
        :param      step_cost:  The step cost
        :type       step_cost:  Int
        """
        actions = [[0,10],[10,0],[10,10],[0,15],[15,0],[15,15],[10,15],[15,10]]
        for a in actions:
            # i = 4-i

            # # Check the states at -60, -30, 0, 30, 60
            
            xn,yn,thetan = plot_curve(x,y,theta,a[0],a[1])
            # Check obstacle condition for the explored nodes 
            if self.isObstacle(xn,yn) == False:
                print("Point :",xn,yn,get_alpha(thetan))
                
                # Check already visited condition for explored node
                if (self.visited_map[int(round(yn*2)),int(round(xn*2)),get_alpha(thetan)])==0:
                    self.visited_map[int(round(yn*2)),int(round(xn*2)),get_alpha(thetan)]=1
                 
                    heapq.heapify(self.queue)
                    heapq.heappush(self.queue,[self.cost((xn,yn),step_cost),xn,yn,thetan,step_cost+1,a,x,y,theta ])

                elif (self.visited_map[int(round(yn*2)),int(round(xn*2)),get_alpha(thetan)])>0:
                    pass

 
        print("-------")

    def backtrack(self):
        n= len(self.visited)
        parent = []
        j = 0
        print("Backtracking")
        print(np.array(self.visited))
        self.shortest_path.append([self.goal[0],self.goal[1],self.goal[2],[0,0]])
        while(True):
            popped = self.visited[n-1-j]

            current_node = [popped[1],popped[2],popped[3],popped[-4]]
            parent_node = [popped[-3],popped[-2],popped[-1],popped[-4]]

            parent.append(parent_node)
           
            self.shortest_path.append([parent_node[0],parent_node[1],parent_node[2],parent_node[3]])
            if [current_node[0],current_node[1]] == [self.start[0],self.start[1]]:
                break
          
            
            # Extract the explored nodes columns of the queue
            cp = np.array(self.visited)[:,1:4]
          

            # Return the index of the parent node in the explored node columns of the queue
            for i in range(0,cp.shape[0]):
                if (cp[i][0]==parent_node[0]) and (cp[i][1]==parent_node[1]) and (cp[i][2]==parent_node[2]):
                  
                    j = n-1-i
      
        sp = np.array(self.shortest_path)
        print("Shortest Path :",sp)
      
        for pt in self.shortest_path:
            plot_curve(pt[0],pt[1],pt[2],pt[3][0],pt[3][1],c='red')
            plt.pause(0.05)

        if plt.waitforbuttonpress():
            sys.exit()
     
    def astar(self):
        """
        A Star Alorithm
        """
        heapq.heapify(self.queue)
        heapq.heappush(self.queue,[self.cost(self.start[:2],0),self.start[0],self.start[1],self.start[2],0,[0,0],self.start[0],self.start[1],self.start[2]])
        while True:

            # Pop the element with least cost
            current = heapq.heappop(self.queue)
            self.visited.append(current)
            print("current_node", np.array(current))

            # Check Goal Condition
            if (np.linalg.norm(np.array(current[1:3])- np.array(self.goal[0:2])) <= 5):
                print("Goal Reached! ")
               
                 
                # Perform backtracking
                self.backtrack()
                break 

            # Search for the available actions in the popped element of the queue
            self.actionsAvailable(current[1],current[2],current[3],current[4])
            print("|________________|")  

            # Represent that node in the animation map
         
           
            plt.show()
            plt.pause(0.005)

          

def cart2img(node):
    return([node[0]+510,node[1]+510,node[2]])

def main():
    
    print("Let the origin be at the center of the Map!")

    start = []
    print("Enter start node in the format [x,y,theta] in (cms,cms,deg) format (Press enter after passing each element):")
    for i in range(0,3):
        x = input()
        start.append(int(x))

    goal = []
    print("Enter goal node in the format [x,y,theta] in (cms,cms,deg) format(Press enter after passing each element):")
    for i in range(0,3):
        x = input()
        goal.append(int(x))

    print("Start: ",start)
    print("Goal: ",goal)


    m = Map(start,goal)

    if m.isObstacle(start[0],start[1]):
        print("ERROR! Start Node is in the obstacle!")
        sys.exit()
    if m.isObstacle(goal[0],goal[1]):
        print("ERROR! Goal Node is in the obstacle!")
        sys.exit()



    m.astar()

if __name__ == "__main__":
    main()
        
