def read_and_modify_file():
    # Ask the user for the filename to read
    input_filename = input("Enter the filename to read: ")
    output_filename = input("Enter the filename to write the modified content to: ")

    try:
        # Open the input file and read its content
        with open(input_filename, 'r') as file:
            content = file.read()

        # Modify the content (e.g., converting text to uppercase)
        modified_content = content.upper()

        # Write the modified content to the output file
        with open(output_filename, 'w') as file:
            file.write(modified_content)

        print(f"The file has been modified and saved as {output_filename}.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' does not exist.")
    except IOError:
        print(f"Error: There was an issue reading or writing to the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Call the function to execute the program
read_and_modify_file()
