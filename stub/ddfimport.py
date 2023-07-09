###############################################################################
# MIT License
# 
# Copyright (c) 2023 The Nature Conservancy - Brazil
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

# This module allows you to import any module from ddf_common.
# To use this module, see the example in sourcepane.ipynb in the parent folder
from IPython.core.display import struct
from ipywidgets import interact_manual
from ipywidgets import Text, Checkbox
import os
import subprocess
import sys
import shutil

# Helper to execute a shell command and print output.
# Returns True if successfully executed.
def shellcmd(batcmd: str) -> bool:
  try:
    print(subprocess.check_output(batcmd, stderr=subprocess.STDOUT, 
                                     shell=True))
  except subprocess.CalledProcessError as e:
    print(e.output)
    return False
  return True

# Pushes to the cloned branch with the given token.
def ddf_push_branch(token:str):
  print('executing ddf_push_branch...')
  try:
    os.chdir(checked_out_path)
  except NameError:
    print(f"Branch not checked out for editing!")
    return

  shellcmd(f'git push https://{token}@github.com/tnc-br/ddf_common.git test')
  os.chdir("/content")

# performs a git add . and commits with the given message.
def ddf_commit_branch(commit_message: struct):
  print('executing ddf_commit_branch...')
  if (len(commit_message) == 0):
    print('No commit message provided.')
    return
  try:
    os.chdir(checked_out_path)
  except NameError:
    print(f"Branch not checked out for editing!")
    return

  if not shellcmd(f'git add .'):
    os.chdir("/content")
    return
  if not shellcmd(f'git commit -m "{commit_message}"'):
    os.chdir("/content")
    return
  print(f"commited change.")
  os.chdir("/content")
  ddf_push_pane()

# if email and branch are not specified, then a simple git clone is performed
# into the tmp folder from the main branch with the expectation that no
# changes are being made. This is the simplest option and considered default.
#
# If branch_name is set, email should also be set. In this case, google drive
# is mounted under content/gdrive and then a folder is created for the branch
# and a git clone for that branch is performed within that folder.
# Since this is on your personal google drive, the files are never lost if the
# colab disappears.
# If you chose a branch name, than a UI appears to commit.
# After you commit, a UI appears to push, which requires a classic github token.
def ddf_import_common(email:str = "", branch_name:str = ""):
  print(f'executing checkout_branch {branch_name}...')
  #branch_name = "" #@param {type:"string"}
  global checked_out_branch
  global checked_out_path
  try:
    print(f"Branch {checked_out_branch} already checked out.")
    print(f"Remember to reload your imports with `importlib.reload(module)`.")
  except NameError:
    checked_out_branch = branch_name if branch_name != "" else "main"
    checked_out_path = ""

  if (len(checked_out_path) > 0):
    sys.path.remove(checked_out_path)
  if branch_name == "":
    if os.path.isdir('/tmp/ddf_common'):
      shutil.rmtree( '/tmp/ddf_common' )
    os.chdir("/tmp")
    if not shellcmd(f'git clone --quiet https://github.com/tnc-br/ddf_common.git || echo "Repository already exists."'):
      os.chdir("/content")
      return
    os.chdir("/content")
    checked_out_branch = "main"
    checked_out_path = '/tmp/ddf_common'
    if checked_out_path not in sys.path:
      sys.path.insert(0, checked_out_path)
    print('main branch checked out as readonly. You may now use ddf_common imports')
  else:
    if (len(email) == 0):
      print('No email provided.')
      return
    if not os.path.isdir("/content/gdrive"):
      from google.colab import drive
      drive.mount("/content/gdrive")

    os.makedirs(f"/content/gdrive/MyDrive/{branch_name}", exist_ok=True)
    os.chdir(f"/content/gdrive/MyDrive/{branch_name}")
    if not shellcmd(f'git clone -b {branch_name} --quiet https://github.com/tnc-br/ddf_common.git || echo "Repository already exists."'):
      os.chdir(f"/content")
      return
    os.chdir(f"/content/gdrive/MyDrive/{branch_name}/ddf_common")
    shellcmd(f'git pull')
    shellcmd(f'git config --global user.email {email}')
    os.chdir(f"/content")
    checked_out_branch = branch_name
    checked_out_path = f'/content/gdrive/MyDrive/{branch_name}/ddf_common'
    if checked_out_path not in sys.path:
      sys.path.insert(0,checked_out_path)
    print(f'{checked_out_branch} branch checked out at "{checked_out_path}". You may now use ddf_common imports and change common files.')
    ddf_commit_pane()

# The entry point for showing the source control pane that allows a git clone,
# commit and push for changes to ddf_common.
def ddf_source_control_pane():
  im = interact_manual.options(manual_name="Checkout Code")
  im(ddf_import_common,
     branch_name = Text(value="",placeholder="Enter branch name",
                      description="Branch name",disabled=False),
     email = Text(value="",placeholder="Enter email",
            description="Email",disabled=False)
  )

# Dynamically displayed commit pane
def ddf_commit_pane():
  interact_manual.options(manual_name="Commit All Changes")(
      ddf_commit_branch,
      commit_message = Text(value="required",placeholder="Enter commit message",
            description="Commit Msg",disabled=False)
  )

# Dynamically displayed push pane
def ddf_push_pane():
  interact_manual.options(manual_name="Push Code")(
      ddf_push_branch,
      push=Checkbox(value=False,
                    description="Push to " + checked_out_branch,disabled=False),
      token = Text(value="required",placeholder="Enter token",
            description="Token",disabled=False)
  )
