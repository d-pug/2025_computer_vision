# Demo: Introduction to Computer Vision 

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

_Maintainer: Nick Ulle <<naulle@ucdavis.edu>>_  

* [Reader](https://ucdavisdatalab.github.io/YOUR_REPOSITORY/)


## Contributing

The workshop reader is a live webpage, hosted through GitHub, where you can
enter curriculum content and post it to a public-facing site for learners.

To make alterations to the reader:

1.  Check in with the reader's current maintainer and notify them about your 
    intended changes. Maintainers might ask you to open an issue, use pull 
    requests, tag your commits with versions, etc.

2.  Run `git pull`, or if it's your first time contributing, see
    [Setup](#setup).

3.  Edit an existing chapter file or create a new one. Chapter files may be 
    either Markdown files (`.md`) or Jupyter Notebook files (`.ipynb`). Either 
    is fine, but you must remain consistent across the reader (i.e. don't mix 
    and match filetypes). Put all chapter files in the `chapters/` directory.
    Enter your text, code, and other information directly into the file. Make 
    sure your file:

    - Follows the naming scheme `##_topic-of-chapter.md/ipynb` (the only 
      exception is `index.md/ipynb`, which contains the reader's front page).
    - Begins with a first-level header (like `# This`). This will be the title
      of your chapter. Subsequent section headers should be second-level
      headers (like `## This`) or below.

    Put any supporting resources in `data/` or `images/`.

4.  Run the command `pixi run rebuild` in this repository to generate HTML
    files for the reader in the `_build/` directory. While developing the
    reader, you can instead run `pixi run build` to only generate HTML for
    parts of the reader that differ from what's in the `_build/` directory.

5.  When you're finished, `git add`:
    - Any files you edited directly
    - Any supporting media you added to `images/`

    Then `git commit` and `git push`. This updates the `main` branch of the
    repo, which contains source materials for the web page (but not the web
    page itself).

6.  Run the following command in a shell at the top level of the repo to update
    the `gh-pages` branch:
    ```
    pixi run publish
    ```
    The live web page will update automatically after 1-10 minutes.

[ghp-import]: https://github.com/c-w/ghp-import


## Setup

We *strongly recommend* using [Pixi][], a fast package manager based on the
conda ecosystem, to install the packages required to build this reader. To
install Pixi, follow [the official instructions][Pixi]. If you prefer not to
use Pixi, it's also possible to manually install the packages using conda or
mamba.

[Pixi]: https://pixi.sh/

The `pixi.toml` file in this repo lists required packages, while the
`pixi.lock` file lists package versions for each platform. When the lock file
is present, Pixi will attempt to install the exact versions listed. Deleting
the lock file allows Pixi to install other versions, which might help if
installation fails (but beware of inconsistencies between package versions).

To install the required packages, open a terminal and navigate to this repo's
directory. Then run:

```sh
pixi install
```

This will automatically create a virtual environment and install the packages.

To open a shell in the virtual environment, run:

```sh
pixi shell
```

You can run the `pixi shell` command from the repo directory or any of its
subdirectories. Use the virtual environment to run any commands related to
building the reader. When you're finished using the virtual environment, you
can use the `exit` command to exit the shell.

> [!NOTE]
> If you're using Windows and Git Bash, the `pixi shell` command is [not yet
> supported][pixi-shell-win]. Instead, you can use the `pixi run` command to
> run commands in the virtual environment. See [the pixi
> documentation][pixi-basics] for examples of how to use `pixi run`.

[pixi-shell-win]: https://github.com/prefix-dev/pixi/issues/417
[pixi-basics]: https://pixi.sh/latest/basic_usage/
