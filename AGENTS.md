Try to avoid adding a flag for every single feature we add. If we're going to use a feature every time, like adding `torch.compile`, we should just
do `torch.compile` every time instead of adding a specific `--compile` flag. Now, if there's something that's actually configurable, like
`--epochs`, then that should be a flag as we will want to change that as we are testing.