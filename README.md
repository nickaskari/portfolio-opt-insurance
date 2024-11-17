# Models for Portfolio Optimization in Insurance

Some cool readMe about the models we are about to create...

Just for reference:
Insane GA library: https://esa.github.io/pygmo/index.html

## Setting up Virtual Environment

```sh
python3.10 -m venv packages 
```
Then activating (MAC OS)..
```sh
source packages/bin/activate
```
For Windows..
```sh
.\packages\Scripts\activate
```
## Handling packages
Add to requirements file after pip installing,
```sh
pip freeze > requirements.txt
```
Installing all libraries from requirements.txt,
```sh
pip install -r requirements.txt
```
## Handling Google Cloud - SSH

## Using `screen` to Run Long-Running Processes

This guide explains how to use `screen` to run long-running Python scripts on a remote server, ensuring they continue running even if you disconnect from your SSH session.

### Step 1: Start a New `screen` Session
Create a new `screen` session with a custom name (e.g., `myproject`):

```bash
screen -S myproject
```

---

### Step 2: Run Your Python Script
Once inside the `screen` session, run your Python script:

```bash
python your_script.py
```

- This will start running your script inside the `screen` session.
- The process will continue running even if you detach from the session.

---

### Step 3: Detach from the `screen` Session
To detach from the session (leave it running in the background):

1. Press `Ctrl + A`, then `D`.

---

### Step 4: List All Active `screen` Sessions
To check which `screen` sessions are currently running:

```bash
screen -ls
```

---

### Step 5: Reattach to a `screen` Session
To reattach to a running session:

```bash
screen -r myproject
```

If you don’t remember the session name, use the session ID:

```bash
screen -r 12345
```

---

### Step 6: Exit a `screen` Session
To stop the process and close the session:

1. Reattach to the session:

   ```bash
   screen -r myproject
   ```

2. Press `Ctrl + C` to stop the script.

3. Type:

   ```bash
   exit
   ```

This will completely close the `screen` session.

---

### Summary of Useful `screen` Commands

| Command                             | Description                                    |
|-------------------------------------|------------------------------------------------|
| `screen -S mysession`               | Start a new screen session named `mysession`.  |
| `Ctrl + A`, `D`                     | Detach from the current screen session.        |
| `screen -ls`                        | List all active screen sessions.               |
| `screen -r mysession`               | Reattach to the named session `mysession`.     |
| `exit`                              | Exit the screen session completely.            |
| `screen -X -S mysession quit`       | Forcefully kill a screen session.              |

---
