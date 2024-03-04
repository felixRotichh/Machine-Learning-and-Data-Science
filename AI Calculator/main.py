from chatterbot import ChatBot

#creating chatbot instance
#using mathematical evaluation logic

Bot = ChatBot(name='Calculator',
              read_on=True,
              logic_adapters=["chatterbot.logic.MathematicalEvaluation"],
              storage_adapter="chatterbot.storage.SQLStorageAdapter")

#clear the screen and start the calculator
print('\033c')
print("Welcome")

while (True):
    #take user input from user
    user_input = input("me: ")

    #check if user typed q to exit program
    if user_input.lower() == 'q' or user_input.lower() == 'quit':
        print("Exiting")
        break

    # otherwise, evaluate the user input
    # print invalid input if the AI is unable to comprehend the input
    try:
        response = Bot.get_response(user_input)
        print("Calculator:", response)
    except:
        print("Calculator: Please enter valid input.")