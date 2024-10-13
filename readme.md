### README.md for Thunder Compute Tests

#### Overview
This project involves testing TensorFlow/Keras custom layers on a Thunder Compute instance. Follow the steps below to set up your environment and execute the tests.

#### Setup and Execution Steps

1. **Connect to Thunder Compute:**
   - Ensure you have set up your token or authentication with Thunder Compute.

2. **Create an Instance:**
   - Log into your Thunder Compute dashboard and create a new instance.

3. **Install SSH Extension in VS Code:**
   - Open Visual Studio Code and install the "Remote - SSH" extension from the marketplace.

4. **Connect VS Code to Running Instance via SSH:**
   - Use the Remote - SSH extension to connect to your instance by entering its SSH details.

5. **Install Git on the Instance:**
   - Once connected, open a terminal in VS Code and install Git:
     ```
     sudo apt-get install git
     ```

6. **Clone the Repository and Install Dependencies:**
   - Clone this repository into your instance:
     ```
     git clone <repository-url>
     cd thunder_compute_tests
     ```
   - Install required Python packages:
     ```
     pip install -r requirements.txt
     ```

7. **Run the Tests:**
   - Execute the test suite by running:
     ```
     python run_tests.py
     ```

#### Additional Information
- Ensure all commands are executed within the Thunder Compute instance environment to maintain consistency and utilize the provided computational resources effectively.
