from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
import pyfiglet


def main():
    welcome = pyfiglet.figlet_format("Algo    CLI", font="basic", width=100)
    print(welcome)
    print("Welcome to Algo CLI - A Data Structures and Algorithms CLI reference.")
    print("Use the arrow keys to choose a menu option, Press Enter to select.")

    while True:
        select_item = select()
        if select_item == "Search":
            search()
        if select_item == "Algorithms":
            algorithms()
        if select_item == "Data Structures":
            data_structures()
        if select_item == None:
            break
    quit()

def select():
    select = inquirer.select(
        message = "Please select an option: \n",
        choices=[
            "Search",
            "Algorithms",
            "Data Structures",
            Choice(value=None, name="Exit")
        ],
        default = "Algorithms",
    ).execute()
    return select

def search():
    search_name = inquirer.text(message = "Enter Algorithm or Data Structure name: ",).execute()
    search_input = input()
    if input == None:
        print("Not found, Please try again")
    else:
        select = inquirer.select(
            message = "Not found, Do you want to try again?",
            choices = [
                "Search Again\n",
                "Go Back",
                "Quit",
            ],
            default = "Search Again",
        ).execute()
        if select == "Search Again":
            search()
        if select == "Go Back":
            main()
        if select == "Quit":
            quit()

def algorithms():
    select = inquirer.select(
        message = "Please choose a Algorithm type: \n",
        choices = [
            "Searching",
            "Sorting",
            "Graphs",
            "Trees",
            "Other\n",
            "Go Back",
            "Quit",
        ],
        default = "Searching",
    ).execute()

    if select == "Searching":
        searching()
    if select == "Sorting":
        sorting()
    if select == "Graphs":
        graphs()
    if select == "Trees":
        trees()
    if select == "Other":
        other()
    if select == "Go Back":
        main() 
    if select == "Quit":
        quit()

def sorting():
    select = inquirer.select(
        message = "Please choose a sorting Algorithm: \n",
        choices = [
            "Insertion Sort",
            "Heap Sort",
            "Selection Sort",
            "Merge Sort",
            "Quick Sort",
            "Couting Sort",
            "Bubble Sort",
            "Radix Sort",
            "Shell Sort",
            "Comb Sort",
            "Pigeonhole Sort",
            "Cycle Sort\n",
            "Go Back",
            "Quit",
        ],
        default = "Searching",
    ).execute()

    if select == "Bubble Sort":
        print(
        "def bubbleSort(arr):\n    n = len(arr)\n      for i in range(n):\n          for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]")
        print("\n Source: https://www.geeksforgeeks.org/bubble-sort/ \n")
    if select == "Go Back":
        main()
    if select == "Quit":
        quit() 

def data_structures():
    select = inquirer.select(
        message = "Please choose a Data Structure: \n",
        choices = [
            "Arrays",
            "Queues",
            "Stacks",
            "Linked Lists",
            "Trees",
            "Graphs\n",
            "Go Back",
            "Quit",
        ],
        default = "Array",
    ).execute()

    if select == "Arrays":
        arrays()
    if select == "Queues":
        queues()
    if select == "Stacks":
        stacks()
    if select == "Linked Lists":
        linked_lists()
    if select == "Trees":
        trees()
    if select == "Graphs":
        Graphs()
    if select == "Go Back":
        main()
    if select == "Quit":
        quit()   

        
if __name__ == "__main__":
    main()