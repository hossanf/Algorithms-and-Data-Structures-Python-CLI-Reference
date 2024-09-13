import java.util.HashMap;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

public class ZeroMQServer {

    public static void main(String[] args) {
        ZMQ.Context context = ZMQ.context(1);
        ZMQ.Socket socket = context.socket(ZMQ.REP);
        socket.bind("tcp://*:5555");
        System.out.println("Connected to Socket- tcp://*:5555, Microservice Running!");

        // Dictionary store of all data to be retrieved
        HashMap<String, Algorithm> algorithms = new HashMap<>();
        algorithms.put("BINARY SEARCH", new BinarySearch());
        algorithms.put("LINEAR SEARCH", new LinearSearch());
        algorithms.put("DEPTH FIRST SEARCH", new DepthFirstSearch());
        algorithms.put("BREADTH FIRST SEARCH", new BreadthFirstSearch());
        algorithms.put("INSERTION SORT", new InsertionSort());
        algorithms.put("HEAP SORT", new HeapSort());
        algorithms.put("SELECTION SORT", new SelectionSort());
        algorithms.put("MERGE SORT", new MergeSort());
        algorithms.put("QUICK SORT", new QuickSort());
        algorithms.put("HUFFMAN CODE", new HuffmanCode());

        HashMap<String, DataStructure> dataStructures = new HashMap<>();
        dataStructures.put("STACK", new Stack());
        dataStructures.put("QUEUE", new Queue());
        dataStructures.put("DEQUE", new Deque());
        dataStructures.put("LINKED LIST", new LinkedList());
        dataStructures.put("HASH TABLE", new HashTable());
        dataStructures.put("KRUSKAL'S ALGORITHM", new KruskalsAlgorithm());
        dataStructures.put("DIJKSTRA'S ALGORITHM", new DijkstrasAlgorithm());
        dataStructures.put("BINARY TREE", new BinaryTree());
        dataStructures.put("AVL TREE", new AVLTree());

        // Search Function
        while (true) {
            byte[] message = socket.recv(0);
            if (message != null) {
                String msg = new String(message);
                System.out.println(msg);
                if (msg.startsWith("SEARCH_")) {
                    String search_term = msg.substring(7).toUpperCase();
                    Algorithm algorithm = algorithms.get(search_term);
                    DataStructure dataStructure = dataStructures.get(search_term);
                    if (algorithm != null) {
                        socket.send(algorithm.run().getBytes(), 0);
                    } else if (dataStructure != null) {
                        socket.send(dataStructure.run().getBytes(), 0);
                    } else {
                        socket.send("NOT_FOUND".getBytes(), 0);
                    }
                } else {
                    Algorithm algorithm = algorithms.get(msg);
                    DataStructure dataStructure = dataStructures.get(msg);
                    if (algorithm != null) {
                        socket.send(algorithm.run().getBytes(), 0);
                    } else if (dataStructure != null) {
                        socket.send(dataStructure.run().getBytes(), 0);
                    } else {
                        socket.send("NOT_FOUND".getBytes(), 0);
                    }
                }
            }
        }
      
    }
    
    

    interface Algorithm {
        String run();
    }

    interface DataStructure {
        String run();
    }

    // Import your algorithm and data structure implementations here
    static class BinarySearch implements Algorithm {
        public String run() {
            
        	return  """
        			
    # Binary Search implementation in Python using a list    			
        			
	def binarySearch(arr, l, r, x):
	    while l <= r:
	
	        mid = l + (r - l) // 2
	
	        # Check if x is present at mid
	        if arr[mid] == x:
	            return mid
	
	        # If x is greater, ignore left half
	        elif arr[mid] < x:
	            l = mid + 1
	
	        # If x is smaller, ignore right half
	        else:
	            r = mid - 1
	
	    # If we reach here, then the element
	    # was not present
	    return -1
	    
	    # source: https://www.geeksforgeeks.org/binary-search/#""";
        			
        }
    }

    static class LinearSearch implements Algorithm {
        public String run() {
            
            return """
            		   		         		        		
    # Linear Search implementation in Python  
            		
    def search(arr, N, x):
	    for i in range(0, N):
	        if (arr[i] == x):
	            return i
	    return -1
	    
    # source: https://www.geeksforgeeks.org/linear-search/
            		""";
        }
    }
    
    static class DepthFirstSearch implements Algorithm {
        public String run() {
            
            return """
            		 		
    # Depth First Search implementation in Python  
            		
    from collections import defaultdict
 
    # This class represents a directed graph using
    # adjacency list representation
     
     
    class Graph:
     
        # Constructor
        def __init__(self):
     
            # default dictionary to store graph
            self.graph = defaultdict(list)
     
        # function to add an edge to graph
        def addEdge(self, u, v):
            self.graph[u].append(v)
     
        # A function used by DFS
        def DFSUtil(self, v, visited):
     
            visited.add(v)
            print(v, end=' ')
    
            for neighbour in self.graph[v]:
                if neighbour not in visited:
                    self.DFSUtil(neighbour, visited)
     
        # The function to do DFS traversal. It uses
        # recursive DFSUtil()
        def DFS(self, v):
     
            # Create a set to store visited vertices
            visited = set()
     
            # Call the recursive helper function
            # to print DFS traversal
            self.DFSUtil(v, visited)
        
    # source:https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/?ref=gcse
            		""";
        }
    }
    static class BreadthFirstSearch implements Algorithm {
        public String run() {
            
            return """
            				
    # Depth First Search implementation in Python
                		
    from collections import defaultdict
 
 
    # This class represents a directed graph
    # using adjacency list representation
    class Graph:
     
        # Constructor
        def __init__(self):
     
            # Default dictionary to store graph
            self.graph = defaultdict(list)
     
        def addEdge(self, u, v):
            self.graph[u].append(v)
     
        # Function to print a BFS of graph
        def BFS(self, s):
     
            # Mark all the vertices as not visited
            visited = [False] * (max(self.graph) + 1)
     
            # Create a queue for BFS
            queue = []
     
            # Mark the source node as
            # visited and enqueue it
            queue.append(s)
            visited[s] = True
     
            while queue:
     
                # Dequeue a vertex from
                # queue and print it
                s = queue.pop(0)
                print(s, end=" ")
     
                # Get all adjacent vertices of the
                # dequeued vertex s. If a adjacent
                # has not been visited, then mark it
                # visited and enqueue it
                for i in self.graph[s]:
                    if visited[i] == False:
                        queue.append(i)
                        visited[i] = True
                    
    # source: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
            		""";
        }
    }
    
    
    static class InsertionSort implements Algorithm {
        public String run() {
            
            return """
            		
    # Insertion Sort implementation in Python            		
            		
    def insertionSortRecursive(arr,n):
	    # base case
	    if n<=1:
	        return
	      
	    # Sort first n-1 elements
	    insertionSortRecursive(arr,n-1)
	    '''Insert last element at its correct position
	        in sorted array.'''
	    last = arr[n-1]
	    j = n-2
	      
	      # Move elements of arr[0..i-1], that are
	      # greater than key, to one position ahead
	      # of their current position 
	    while (j>=0 and arr[j]>last):
	        arr[j+1] = arr[j]
	        j = j-1
	  
	    arr[j+1]=last
	    
    # source: https://www.geeksforgeeks.org/recursive-insertion-sort """;
        }
    }
    
    static class HeapSort	 implements Algorithm {
        public String run() {
            
            return """
            		
    # Heap Sort implementation in Python
            		            		
    def heapify(arr, N, i):
	    largest = i  # Initialize largest as root
	    l = 2 * i + 1     # left = 2*i + 1
	    r = 2 * i + 2     # right = 2*i + 2
	 
	    # See if left child of root exists and is
	    # greater than root
	    if l < N and arr[largest] < arr[l]:
	        largest = l
	 
	    # See if right child of root exists and is
	    # greater than root
	    if r < N and arr[largest] < arr[r]:
	        largest = r
	 
	    # Change root, if needed
	    if largest != i:
	        arr[i], arr[largest] = arr[largest], arr[i]  # swap
	 
	        # Heapify the root.
	        heapify(arr, N, largest)
	 
    # The main function to sort an array of given size
    def heapSort(arr):
        N = len(arr)
     
        # Build a maxheap.
        for i in range(N//2 - 1, -1, -1):
            heapify(arr, N, i)
     
        for i in range(N-1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]  # swap
            heapify(arr, i, 0)
            
    # source:https://www.geeksforgeeks.org/heap-sort""";
        }
    }
    
    static class SelectionSort	 implements Algorithm {
        public String run() {
            
            return """
            		
    # Selection Sort implementation in Python            		        		
            		
    # Return minimum index
    def minIndex( a , i , j ):
        if i == j:
            return i
             
        # Find minimum of remaining elements
        k = minIndex(a, i + 1, j)
         
        # Return minimum of current
        # and remaining.
        return (i if a[i] < a[k] else k)
         
    # Recursive selection sort. n is
    # size of a[] and index is index of
    # starting element.
    def recurSelectionSort(a, n, index = 0):
     
        # Return when starting and
        # size are same
        if index == n:
            return -1
             
        # calling minimum index function
        # for minimum index
        k = minIndex(a, index, n-1)
         
        # Swapping when index and minimum
        # index are not same
        if k != index:
            a[k], a[index] = a[index], a[k]
             
        # Recursively calling selection
        # sort function
        recurSelectionSort(a, n, index + 1)
        
    # source: https://www.geeksforgeeks.org/recursive-selection-sort""";
        }
    }
    
    static class MergeSort	 implements Algorithm {
        public String run() {
            
            return """
            		
    # Merge Sort implementation in Python             		
            		
    def mergeSort(arr):
	    if len(arr) > 1:
	 
	         # Finding the mid of the array
	        mid = len(arr)//2
	 
	        # Dividing the array elements
	        L = arr[:mid]
	 
	        # into 2 halves
	        R = arr[mid:]
	 
	        # Sorting the first half
	        mergeSort(L)
	 
	        # Sorting the second half
	        mergeSort(R)
	 
	        i = j = k = 0
	 
	        # Copy data to temp arrays L[] and R[]
	        while i < len(L) and j < len(R):
	            if L[i] <= R[j]:
	                arr[k] = L[i]
	                i += 1
	            else:
	                arr[k] = R[j]
	                j += 1
	            k += 1
	 
	        # Checking if any element was left
	        while i < len(L):
	            arr[k] = L[i]
	            i += 1
	            k += 1
	 
	        while j < len(R):
	            arr[k] = R[j]
	            j += 1
	            k += 1
            
    # source: https://www.geeksforgeeks.org/merge-sort/""";
        }
    }
    
    static class QuickSort	 implements Algorithm {
        public String run() {
            
            return """
            		
    # Quick Sort implementation in Python             		
            		
    def partition(arr, low, high):
	    i = (low - 1)         # index of smaller element
	    pivot = arr[high]     # pivot
	  
	    for j in range(low, high):
	  
	        # If current element is smaller 
	        # than or equal to pivot
	        if arr[j] <= pivot:
	          
	            # increment index of
	            # smaller element
	            i += 1
	            arr[i], arr[j] = arr[j], arr[i]
	  
	    arr[i + 1], arr[high] = arr[high], arr[i + 1]
	    return (i + 1)
  
    # The main function that implements QuickSort
    # arr[] --> Array to be sorted,
    # low --> Starting index,
    # high --> Ending index
      
    # Function to do Quick sort
    def quickSort(arr, low, high):
        if low < high:
      
            # pi is partitioning index, arr[p] is now
            # at right place
            pi = partition(arr, low, high)
      
            # Separately sort elements before
            # partition and after partition
            quickSort(arr, low, pi-1)
            quickSort(arr, pi + 1, high)
            
    # source:https://www.geeksforgeeks.org/iterative-quick-sort/""";
        }
    }
    
    static class HuffmanCode	 implements Algorithm {
        public String run() {
            
            return """
            		
# A Huffman Tree Node in Python
import heapq
 
 
class node:
    def __init__(self, freq, symbol, left=None, right=None):
        # frequency of symbol
        self.freq = freq
 
        # symbol name (character)
        self.symbol = symbol
 
        # node left of current node
        self.left = left
 
        # node right of current node
        self.right = right
 
        # tree direction (0/1)
        self.huff = ''
 
    def __lt__(self, nxt):
        return self.freq < nxt.freq
 
 
    # utility function to print huffman
    # codes for all symbols in the newly
    # created Huffman tree
    def printNodes(node, val=''):
     
        # huffman code for current node
        newVal = val + str(node.huff)
     
        # if node is not an edge node
        # then traverse inside it
        if(node.left):
            printNodes(node.left, newVal)
        if(node.right):
            printNodes(node.right, newVal)
     
            # if node is edge node then
            # display its huffman code
        if(not node.left and not node.right):
            print(f"{node.symbol} -> {newVal}")
     
     
    # characters for huffman tree
    chars = ['a', 'b', 'c', 'd', 'e', 'f']
     
    # frequency of characters
    freq = [5, 9, 12, 13, 16, 45]
     
    # list containing unused nodes
    nodes = []
     
    # converting characters and frequencies
    # into huffman tree nodes
    for x in range(len(chars)):
        heapq.heappush(nodes, node(freq[x], chars[x]))
     
    while len(nodes) > 1:
     
        # sort all the nodes in ascending order
        # based on their frequency
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
     
        # assign directional value to these nodes
        left.huff = 0
        right.huff = 1
     
        # combine the 2 smallest nodes to create
        # new node as their parent
        newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right)
     
        heapq.heappush(nodes, newNode)
     
    # Huffman Tree is ready!
    printNodes(nodes[0])

    # source: https://www.geeksforgeeks.org/huffman-coding-greedy-algo-3/""";
        }
    }
    
    
    

    static class Stack implements DataStructure {
        public String run() {
            
        	return  """
        			
    # Stack implementation in Python using a list
     
    stack = []
     
    # append() function to push
    # element in the stack
    stack.append('a')
    stack.append('b')
    stack.append('c')
     
    print('Initial stack')
    print(stack)
     
    # pop() function to pop
    # element from stack in
    # LIFO order
    print('\nElements popped from stack:')
    print(stack.pop())
    print(stack.pop())
    print(stack.pop())
     
    print('\nStack after elements are popped:')
    print(stack)
    
    # source: https://www.geeksforgeeks.org/stack-in-python/""";
        			
        }
    }
    
    
    static class Queue implements DataStructure {
        public String run() {
            
        	return  """
        			
    # Queue implementation in Python using a list
      
    # Initializing a queue
    queue = []
      
    # Adding elements to the queue
    queue.append('a')
    queue.append('b')
    queue.append('c')
      
    print("Initial queue")
    print(queue)
      
    # Removing elements from the queue
    print("\nElements dequeued from queue")
    print(queue.pop(0))
    print(queue.pop(0))
    print(queue.pop(0))
      
    print("\nQueue after removing elements")
    print(queue)
    
    # source: https://www.geeksforgeeks.org/queue-in-python/""";
        			
        }
    }
    
    
    static class Deque implements DataStructure {
        public String run() {
            
        	return  """
        						
    # Deque implementation in Python
    
    # insert(), index(), remove(), count()
    
    import collections
     
    # initializing deque
    de = collections.deque([1, 2, 3, 3, 4, 2, 4])
     
    # using index() to print the first occurrence of 4
    print ("The number 4 first occurs at a position : ")
    print (de.index(4,2,5))
     
    # using insert() to insert the value 3 at 5th position
    de.insert(4,3)
     
    # printing modified deque
    print ("The deque after inserting 3 at 5th position is : ")
    print (de)
     
    # using count() to count the occurrences of 3
    print ("The count of 3 in deque is : ")
    print (de.count(3))
     
    # using remove() to remove the first occurrence of 3
    de.remove(3)
     
    # printing modified deque
    print ("The deque after deleting first occurrence of 3 is : ")
    print (de)
    
    # source: https://www.geeksforgeeks.org/deque-in-python/""";
        			
        }
    }
    
    static class LinkedList implements DataStructure {
        public String run() {
            
        	return  """
        			
    # Linked List implementation in Python         			
        			
    class Node:
       def __init__(self, dataval=None):
          self.dataval = dataval
          self.nextval = None

    class SLinkedList:
       def __init__(self):
          self.headval = None
    
    list1 = SLinkedList()
    list1.headval = Node("Mon")
    e2 = Node("Tue")
    e3 = Node("Wed")
    # Link first Node to second node
    list1.headval.nextval = e2
    
    # Link second Node to third node
    e2.nextval = e3
    
    # source: https://www.tutorialspoint.com/python_data_structure/python_linked_lists.htm""";
        			
        }
    }
    
    static class HashTable implements DataStructure {
        public String run() {
            
        	return  """
        			
    # Simple Hash table in Python using a dictionary
    
    # Declare a dictionary and update value
    dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
    
    # update existing entry
    dict['Age'] = 8;
    
     # Add new entry               
    dict['School'] = "DPS School"; 
    print ("dict['Age']: ", dict['Age'])
    print ("dict['School']: ", dict['School'])
    
    # Delete element 
    dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
    del dict['Name']; # remove entry with key 'Name'
    dict.clear();     # remove all entries in dict
    del dict ;        # delete entire dictionary
    
    print ("dict['Age']: ", dict['Age'])
    print ("dict['School']: ", dict['School'])
    
    # source: https://www.tutorialspoint.com/python_data_structure/python_hash_table.htm""";
        			
        }
    }
    
    
    static class KruskalsAlgorithm implements DataStructure {
        public String run() {
            
        	return  """
        			
    # Kruskal's Algorithm in Python         			
        			
    # Class to represent a graph
    class Graph:
     
        def __init__(self, vertices):
            self.V = vertices
            self.graph = []
     
        # Function to add an edge to graph
        def addEdge(self, u, v, w):
            self.graph.append([u, v, w])
     
        # A utility function to find set of an element i
        # (truly uses path compression technique)
        def find(self, parent, i):
            if parent[i] != i:
     
                # Reassignment of node's parent
                # to root node as
                # path compression requires
                parent[i] = self.find(parent, parent[i])
            return parent[i]
     
        # A function that does union of two sets of x and y
        # (uses union by rank)
        def union(self, parent, rank, x, y):
     
            # Attach smaller rank tree under root of
            # high rank tree (Union by Rank)
            if rank[x] < rank[y]:
                parent[x] = y
            elif rank[x] > rank[y]:
                parent[y] = x
     
            # If ranks are same, then make one as root
            # and increment its rank by one
            else:
                parent[y] = x
                rank[x] += 1
     
        # The main function to construct MST Kruskal's algorithm
        def KruskalMST(self):
     
            # This will store the resultant MST
            result = []
     
            # An index variable, used for sorted edges
            i = 0
     
            # An index variable, used for result[]
            e = 0
     
            # Sort all the edges in
            # non-decreasing order of their
            # weight
            self.graph = sorted(self.graph,
                                key=lambda item: item[2])
     
            parent = []
            rank = []
     
            # Create V subsets with single elements
            for node in range(self.V):
                parent.append(node)
                rank.append(0)
     
            # Number of edges to be taken is less than to V-1
            while e < self.V - 1:
     
                # Pick the smallest edge and increment
                # the index for next iteration
                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)
     
                # If including this edge doesn't
                # cause cycle, then include it in result
                # and increment the index of result
                # for next edge
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.union(parent, rank, x, y)
                # Else discard the edge
     
            minimumCost = 0
            print("Edges in the constructed MST")
            for u, v, weight in result:
                minimumCost += weight
                print("%d -- %d == %d" % (u, v, weight))
            print("Minimum Spanning Tree", minimumCost)
     
     
        # Driver code
    if __name__ == '__main__':
        g = Graph(4)
        g.addEdge(0, 1, 10)
        g.addEdge(0, 2, 6)
        g.addEdge(0, 3, 5)
        g.addEdge(1, 3, 15)
        g.addEdge(2, 3, 4)
     
        # Function call
        g.KruskalMST()
         
        # This code is contributed by Neelam Yadav
        # Improved by James Gra�a-Jones
        # source: https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/""";
        			
        }
    }
    
    
    static class DijkstrasAlgorithm implements DataStructure {
        public String run() {
            
        	return  """
        			
    # Dijkstra's Algorithm in Python for single source shortest path.
    # The program is for adjacency matrix representation of the graph
      
    # Library for INT_MAX
    import sys
      
      
    class Graph():
      
        def __init__(self, vertices):
            self.V = vertices
            self.graph = [[0 for column in range(vertices)]
                          for row in range(vertices)]
      
        def printSolution(self, dist):
            print("Vertex \tDistance from Source")
            for node in range(self.V):
                print(node, "\t", dist[node])
      
        # A utility function to find the vertex with
        # minimum distance value, from the set of vertices
        # not yet included in shortest path tree
        def minDistance(self, dist, sptSet):
      
            # Initialize minimum distance for next node
            min = sys.maxsize
      
            # Search not nearest vertex not in the
            # shortest path tree
            for u in range(self.V):
                if dist[u] < min and sptSet[u] == False:
                    min = dist[u]
                    min_index = u
      
            return min_index
      
        # Function that implements Dijkstra's single source
        # shortest path algorithm for a graph represented
        # using adjacency matrix representation
        def dijkstra(self, src):
      
            dist = [sys.maxsize] * self.V
            dist[src] = 0
            sptSet = [False] * self.V
      
            for cout in range(self.V):
      
                # Pick the minimum distance vertex from
                # the set of vertices not yet processed.
                # x is always equal to src in first iteration
                x = self.minDistance(dist, sptSet)
      
                # Put the minimum distance vertex in the
                # shortest path tree
                sptSet[x] = True
      
                # Update dist value of the adjacent vertices
                # of the picked vertex only if the current
                # distance is greater than new distance and
                # the vertex in not in the shortest path tree
                for y in range(self.V):
                    if self.graph[x][y] > 0 and sptSet[y] == False and 
                            dist[y] > dist[x] + self.graph[x][y]:
                        dist[y] = dist[x] + self.graph[x][y]
      
            self.printSolution(dist)
      
          
        # Driver's code
        if __name__ == "__main__":
            g = Graph(9)
            g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
                       [4, 0, 8, 0, 0, 0, 0, 11, 0],
                       [0, 8, 0, 7, 0, 4, 0, 0, 2],
                       [0, 0, 7, 0, 9, 14, 0, 0, 0],
                       [0, 0, 0, 9, 0, 10, 0, 0, 0],
                       [0, 0, 4, 14, 10, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 2, 0, 1, 6],
                       [8, 11, 0, 0, 0, 0, 1, 0, 7],
                       [0, 0, 2, 0, 0, 0, 6, 7, 0]
                       ]
          
            g.dijkstra(0)
          
       # This code is contributed by Divyanshu Mehta and Updated by Pranav Singh Sambyal
       # source: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/""";
        			
        }
    }
    
    static class BinaryTree implements DataStructure {
        public String run() {
            
        	return  """	
        				
    # Binary Tree implementation with operations in Python
    # Create Root
    class Node:
       def __init__(self, data):
          self.left = None
          self.right = None
          self.data = data
          
    # Insert
    def insert(self, data):
        # Compare the new value with the parent node
      if self.data:
         if data < self.data:
            if self.left is None:
               self.left = Node(data)
            else:
               self.left.insert(data)
         elif data > self.data:
               if self.right is None:
                  self.right = Node(data)
               else:
                  self.right.insert(data)
      else:
         self.data = data
         
    # Inorder traversal
    # Left -> Root -> Right
   def inorderTraversal(self, root):
      res = []
      if root:
         res = self.inorderTraversal(root.left)
         res.append(root.data)
         res = res + self.inorderTraversal(root.right)
      return res
      
    print(root.inorderTraversal(root)) 

    # source: https://www.tutorialspoint.com/python_data_structure/python_binary_tree.htm""";
        			
        }
    }
    
    
    static class AVLTree implements DataStructure {
        public String run() {
            
        	return  """
        			
   # AVL Tree in Python
   
    class Node(object):
       def __init__(self, data):
          self.data = data
          self.left = None
          self.right = None
          self.height = 1
    class AVLTree(object):
       def insert(self, root, key):
          if not root:
             return Node(key)
          elif key < root.data:
             root.left = self.insert(root.left, key)
          else:
             root.right = self.insert(root.right, key)
          root.h = 1 + max(self.getHeight(root.left),
             self.getHeight(root.right))
          b = self.getBalance(root)
          if b > 1 and key < root.left.data:
             return self.rightRotate(root)
          if b < -1 and key > root.right.data:
             return self.leftRotate(root)
          if b > 1 and key > root.left.data:
             root.left = self.lefttRotate(root.left)
             return self.rightRotate(root)
          if b < -1 and key < root.right.data:
             root.right = self.rightRotate(root.right)
             return self.leftRotate(root)
          return root
       def leftRotate(self, z):
          y = z.right
          T2 = y.left
          
    # source: https://www.tutorialspoint.com/data_structures_algorithms/avl_tree_algorithm.htm""";
        			
        }
    }
    
    
    
    
    
    
    
    
    
}
