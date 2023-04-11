#!/bin/sh
rtdir='/home/data/lwb/code'
expdir='/home/data/lwb/experiments'
expdirname='rain'
cfgdir='/home/data/lwb/config/'${expdirname}
cfgdirname='config'
src='rain' # github
dst='Rain' # gitlab
branch='dev'
commit_detail='update README.md'

cd ${rtdir}/${src}
rm -f ./experiments/${expdirname}
rm -f ./experiments/${cfgdirname}
git checkout ${branch}
cd ../${dst}
git checkout ${branch}
git checkout -b ${branch}
git checkout ${branch}
cd ${rtdir}
mv ${dst}/.git ${dst}_git
rm -rf ${dst}
cp -r ${rtdir}/${src} ${rtdir}/${dst}
rm -rf ${dst}/.git
mv ${dst}_git ${dst}/.git

# for n in ${dst}
for n in ${src} ${dst}
do
    cd ${rtdir}/${n}
    git add .
    git status
    git commit -m "${commit_detail}"
    git push origin -u ${branch}
    git push
done

# git log
# git reset --hard !@#$%^&*()
# git push origin fblr -f

cd ${rtdir}/${dst}
ln -s ${expdir}/${expdirname} ./experiments/${expdirname}
ln -s ${cfgdir}/${cfgdirname} ./experiments/${cfgdirname}
git branch
cd ${rtdir}/${src}
ln -s ${expdir}/${expdirname} ./experiments/${expdirname}
ln -s ${cfgdir}/${cfgdirname} ./experiments/${cfgdirname}
git branch