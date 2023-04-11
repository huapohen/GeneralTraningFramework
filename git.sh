#!/bin/sh
rtdir='/home/data/lwb/code'
expdir='/home/data/lwb/experiments'
expdirname='rain'
src='rain' # github
dst='Rain' # gitlab
branch='main'
commit_detail='Initial Commit'

cd ${rtdir}/${src}
rm -f ./experiments/${expdirname}
git checkout ${branch}
cd ../${dst}
git checkout ${branch}
git checkout -b ${branch}
git checkout ${branch}
cd ${rtdir}
mv ${dst}/.git cp.git
rm -rf ${dst}
cp -r ${rtdir}/${src} ${rtdir}/${dst}
mv cp.git ${dst}/.git

# for n in ${dst}
for n in ${src} ${dst}
do
    cd ${rtdir}/${n}
    git add .
    git status
    git commit -m "${commit_detail}"
    git push
    git push origin -u ${branch}
    git push
done

# git log
# git reset --hard !@#$%^&*()
# git push origin fblr -f

cd ${rtdir}/${dst}
git branch
cd ${rtdir}/${src}
ln -s ${expdir}/${expdirname} ./experiments/${expdirname}
git branch