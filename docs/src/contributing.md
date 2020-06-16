# Contributing

Contributions are welcomed. Here's the workflow for development of new features, refactoring and bugfix.

```
master             # Stable branch, always ready to be tagged
dev (or develop)   # Development branch. New features, refactoring, bug and hotfix are integrated into dev before going into master.
feature/           # New feature needs a `feature` prefix
   struct-eva      # Example of a new feature named `struct-eva`   
refactor/          # Refactoring are tagged with a `refactor` prefix
   struct-eva      # Example of refactoring the `struct-eva` feature
bug/               # Prefix for bugs found during development
   data-fix        # Example where we fix a dataset
hotfix/            # Prefix for hotfix (bugs for deployed versions)
   example-fix     # Example of a bugfix
```
