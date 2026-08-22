# Version Control Workflow — MLOps Iris Classifier

## 1. Overview

This document describes the Git-based version control workflow
used for this Machine Learning project.

- Repository: https://github.com/devendra2645-Tech/mlops-iris-classifier
- Primary language: Python
- Maintainer: Devendra Patil

## 2. Branching Strategy

| Branch | Purpose |
|--------|---------|
| main | Stable, deployable code |
| develop | Integration branch for development |
| feature/<name> | Individual features |
| conflict-demo-* | Conflict resolution practice |

Rule:

feature/* → Pull Request → develop → main

No one should commit directly to main.

## 3. Commit Convention

Commits follow this format:

type: short description

Examples:

feat: add classification report
fix: correct data loading bug
docs: update documentation
chore: update requirements
refactor: improve training code

## 4. Standard Workflow

1. Switch to develop.
2. Pull the latest changes.
3. Create a feature branch.
4. Make changes.
5. Stage the changes.
6. Commit the changes.
7. Push the feature branch.
8. Create a Pull Request.
9. Review and merge the Pull Request into develop.

## 5. Merge Conflict Resolution

1. Attempt the merge.
2. Open the conflicted file.
3. Find the conflict markers.
4. Decide which changes to keep.
5. Remove the conflict markers.
6. Run git add.
7. Commit the resolved changes.
8. Test the project.

## 6. .gitignore Policy

Large files such as datasets, trained models,
virtual environments, and notebook checkpoints
should not be committed directly to Git.

These files can be managed separately using tools
such as DVC or cloud storage.

## 7. Pull Request Checklist

- Code runs without errors.
- No large data/model files are staged.
- Commit messages follow the convention.
- Branch is up to date with develop.
- Pull Request explains what changed and why.

## 8. Lessons Learned

Git helps maintain the history of the ML project,
while branching and Pull Requests help developers
work safely and collaboratively.