---
title: 将hexo搭建的博客迁移到新的环境
date: 2019-04-28 10:58:31
tags: hexo
description: 迁移hexo下搭建的博客，在新的电脑中同步更新博客文件
---

### Abstract

本篇博客主要是介绍如何在新的环境（如新的电脑，新的用户，重装系统之后等）下继续使用hexo同步更新博客内容，即完成博客的迁移。在操作过程中，对于hexo+github搭建博客的一些原理有更深的认识，因此总结一下。



### Procedure

在迁移之前，简单说明一下hexo和github在博客文件管理中是如何分工的。在github上创建的仓库\<username\>.github.io包含了两个分支：blog（自己命名的分支）和master，其中blog分支用来托管博客原始文件包括md文件、图片、主题文件等，是可以编辑改变的文件；master分支用来存放生成的静态网页，可以通过在_config.yml文件中设置deployment，使得hexo的部署自动在master分支上完成。

```yml
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repository: https://github.com/newbieyxy/newbieyxy.github.io.git
  branch: master
```

因此，在这样的管理方法下，只需要将博客原始文件迁移到新的环境下，再进行hexo的配置和依赖的安装（类似于最初搭建博客时的操作），就可以继续更新博客了。

###### Step 1.

安装 git bash，node.js（包括了环境变量的配置和npm的安装）

###### Step 2.

将github上博客仓库保存原始文件的blog分支拷贝到本地，这里补充一点：在刚开始创建博客仓库的时候，可以设置blog分支设为默认分支，那么之后的git add、git commit和git push等操作就更加的方便。拷贝之后保留的文件包括：

 _config.yml
package.json
scaffolds/
source/
themes/

在git bash进入本地仓库目录，使用npm安装各种依赖包，执行的命令如下：

```
npm install
npm install hexo-cli -g
npm install hexo-deploy-git --save
npm install hexo-server --save
```

首先，npm install是根据package.json来读取依赖的信息并安装的，如果package中原来需要的一些依赖包的版本较低（如在安装过程中出现提示“core-js@1.2.7: core-js@<2.6.5 is no longer maintained. Please, upgrade to core-js@3 or at least to actual version of core-js@2.”），则需要再次针对某个包重新安装 (如npm install core-js)；另一种办法是在package文件中修改依赖包的版本号，重新执行npm install命令。

然后是安装hexo，注意是全局安装，否则会在运行hexo时报错“ERROR Local hexo not found in ...”。

因为需要将博客部署到github，因此安装hexo-deploy-git插件包；另外想要在本地预览博客网页，则安装hexo-server。

PS：如果npm过程经常卡主/中断（毕竟使用的是国外服务器），可以在npm的配置文件中增加代理：

```bash
npm config set registry http://registry.cnpmjs.org
```



###### Step 3.

将新的md文件添加到博客仓库的source文件_posts目录下，用hexo g -d生成和部署，此时完成博客的迁移和更新。另外，将仓库提交到github上，完成原始文件的备份。



Ref：

[换了电脑如何使用hexo继续写博客]( https://www.cnblogs.com/study-everyday/p/8902136.html )

