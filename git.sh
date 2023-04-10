#!/bin/sh
rtdir='/home/data/lwb/code'
expdir='/home/data/lwb/experiments'
dst='rain'
branch='rain'
commit_detail=' init commit '

cd ${rtdir}/${dst}
git checkout ${branch}
rm -f ./experiments/${dst}

for n in ${dst}
do
    cd ${rtdir}/${n}
    git add .
    git status
    git commit -m "${commit_detail}"
    # git push
    # git push origin -u ${branch}
    git push
done

# git log
# git reset --hard !@#$%^&*()
# git push origin fblr -f

cd ${rtdir}/${dst}
git branch
ln -s ${expdir}/${dst} ./experiments/${dst}