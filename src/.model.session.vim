let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
imap <S-Tab> <Plug>SuperTabBackward
inoremap <C-Tab> 	
inoremap <silent> <Plug>NERDCommenterInsert  <BS>:call NERDComment('i', "insert")
noremap 	= V)=''    
map  :NERDTreeToggle
nmap o <Plug>ZoomWin
map [B <Down>
nmap ,ca <Plug>NERDCommenterAltDelims
xmap ,cu <Plug>NERDCommenterUncomment
nmap ,cu <Plug>NERDCommenterUncomment
xmap ,cb <Plug>NERDCommenterAlignBoth
nmap ,cb <Plug>NERDCommenterAlignBoth
xmap ,cl <Plug>NERDCommenterAlignLeft
nmap ,cl <Plug>NERDCommenterAlignLeft
nmap ,cA <Plug>NERDCommenterAppend
xmap ,cy <Plug>NERDCommenterYank
nmap ,cy <Plug>NERDCommenterYank
xmap ,cs <Plug>NERDCommenterSexy
nmap ,cs <Plug>NERDCommenterSexy
xmap ,ci <Plug>NERDCommenterInvert
nmap ,ci <Plug>NERDCommenterInvert
nmap ,c$ <Plug>NERDCommenterToEOL
xmap ,cn <Plug>NERDCommenterNested
nmap ,cn <Plug>NERDCommenterNested
xmap ,cm <Plug>NERDCommenterMinimal
nmap ,cm <Plug>NERDCommenterMinimal
xmap ,c  <Plug>NERDCommenterToggle
nmap ,c  <Plug>NERDCommenterToggle
xmap ,cc <Plug>NERDCommenterComment
nmap ,cc <Plug>NERDCommenterComment
map ,rwp <Plug>RestoreWinPosn
map ,swp <Plug>SaveWinPosn
map ,tt <Plug>AM_tt
map ,tsq <Plug>AM_tsq
map ,tsp <Plug>AM_tsp
map ,tml <Plug>AM_tml
map ,tab <Plug>AM_tab
map ,m= <Plug>AM_m=
map ,tW@ <Plug>AM_tW@
map ,t@ <Plug>AM_t@
map ,t~ <Plug>AM_t~
map ,t? <Plug>AM_t?
map ,w= <Plug>AM_w=
map ,ts= <Plug>AM_ts=
map ,ts< <Plug>AM_ts<
map ,ts; <Plug>AM_ts;
map ,ts: <Plug>AM_ts:
map ,ts, <Plug>AM_ts,
map ,t= <Plug>AM_t=
map ,t< <Plug>AM_t<
map ,t; <Plug>AM_t;
map ,t: <Plug>AM_t:
map ,t, <Plug>AM_t,
map ,t# <Plug>AM_t#
map ,t| <Plug>AM_t|
map ,T~ <Plug>AM_T~
map ,Tsp <Plug>AM_Tsp
map ,Tab <Plug>AM_Tab
map ,TW@ <Plug>AM_TW@
map ,T@ <Plug>AM_T@
map ,T? <Plug>AM_T?
map ,T= <Plug>AM_T=
map ,T< <Plug>AM_T<
map ,T; <Plug>AM_T;
map ,T: <Plug>AM_T:
map ,Ts, <Plug>AM_Ts,
map ,T, <Plug>AM_T,o
map ,T# <Plug>AM_T#
map ,T| <Plug>AM_T|
map ,Htd <Plug>AM_Htd
map ,anum <Plug>AM_aunum
map ,aenum <Plug>AM_aenum
map ,aunum <Plug>AM_aunum
map ,afnc <Plug>AM_afnc
map ,adef <Plug>AM_adef
map ,adec <Plug>AM_adec
map ,ascom <Plug>AM_ascom
map ,aocom <Plug>AM_aocom
map ,adcom <Plug>AM_adcom
map ,acom <Plug>AM_acom
map ,abox <Plug>AM_abox
map ,a( <Plug>AM_a(
map ,a= <Plug>AM_a=
map ,a< <Plug>AM_a<
map ,a, <Plug>AM_a,
map ,a? <Plug>AM_a?
nmap ,ihn :IHN
nmap ,is :IHS
nmap ,ih :IHS
nnoremap ,gd :Gdiff
map ,s? z=
map ,sa zg
map ,ssfr :setlocal spell! spelllang=fr
map ,ss :setlocal spell!
map ,ht :AT
map ,h :AV
noremap ,e :! ./%
noremap ,ct :tab split
noremap ,mctags :!/usr/bin/ctags -R --c++-kinds=+p --fields=+iaS --extra=+q .
noremap ,tup :TlistUpdate
noremap ,tb :TagbarToggle
noremap ,tl :TlistToggle
noremap ,to :TlistAddFiles %
nnoremap ; : 
map Q gq
xmap S <Plug>VSurround
onoremap <silent> [L :call NextIndent(1, 0, 1, 1)
onoremap <silent> [l :call NextIndent(0, 0, 0, 1)
vnoremap <silent> [L :call NextIndent(0, 0, 1, 1)
vnoremap <silent> [l :call NextIndent(0, 0, 0, 1)
nnoremap <silent> [L :call NextIndent(0, 0, 1, 1)
nnoremap <silent> [l :call NextIndent(0, 0, 0, 1)
onoremap <silent> ]L :call NextIndent(1, 1, 1, 1)
onoremap <silent> ]l :call NextIndent(0, 1, 0, 1)
vnoremap <silent> ]L :call NextIndent(0, 1, 1, 1)
vnoremap <silent> ]l :call NextIndent(0, 1, 0, 1)
nnoremap <silent> ]L :call NextIndent(0, 1, 1, 1)
nnoremap <silent> ]l :call NextIndent(0, 1, 0, 1)
nmap cS <Plug>CSurround
nmap cs <Plug>Csurround
nmap ds <Plug>Dsurround
nmap <silent> da diWa
nmap <silent> dw diwi
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
xmap gS <Plug>VgSurround
nnoremap tc :TagbarShowTag
nnoremap tf :TlistShowTag
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(expand((exists("g:netrw_gx")? g:netrw_gx : '<cfile>')),netrw#CheckIfRemote())
nnoremap <SNR>58_: :=v:count ? v:count : ''
vnoremap <silent> <Plug>(calendar) :Calendar
nnoremap <silent> <Plug>(calendar) :Calendar
xnoremap <silent> <Plug>NERDCommenterUncomment :call NERDComment("x", "Uncomment")
nnoremap <silent> <Plug>NERDCommenterUncomment :call NERDComment("n", "Uncomment")
xnoremap <silent> <Plug>NERDCommenterAlignBoth :call NERDComment("x", "AlignBoth")
nnoremap <silent> <Plug>NERDCommenterAlignBoth :call NERDComment("n", "AlignBoth")
xnoremap <silent> <Plug>NERDCommenterAlignLeft :call NERDComment("x", "AlignLeft")
nnoremap <silent> <Plug>NERDCommenterAlignLeft :call NERDComment("n", "AlignLeft")
nnoremap <silent> <Plug>NERDCommenterAppend :call NERDComment("n", "Append")
xnoremap <silent> <Plug>NERDCommenterYank :call NERDComment("x", "Yank")
nnoremap <silent> <Plug>NERDCommenterYank :call NERDComment("n", "Yank")
xnoremap <silent> <Plug>NERDCommenterSexy :call NERDComment("x", "Sexy")
nnoremap <silent> <Plug>NERDCommenterSexy :call NERDComment("n", "Sexy")
xnoremap <silent> <Plug>NERDCommenterInvert :call NERDComment("x", "Invert")
nnoremap <silent> <Plug>NERDCommenterInvert :call NERDComment("n", "Invert")
nnoremap <silent> <Plug>NERDCommenterToEOL :call NERDComment("n", "ToEOL")
xnoremap <silent> <Plug>NERDCommenterNested :call NERDComment("x", "Nested")
nnoremap <silent> <Plug>NERDCommenterNested :call NERDComment("n", "Nested")
xnoremap <silent> <Plug>NERDCommenterMinimal :call NERDComment("x", "Minimal")
nnoremap <silent> <Plug>NERDCommenterMinimal :call NERDComment("n", "Minimal")
xnoremap <silent> <Plug>NERDCommenterToggle :call NERDComment("x", "Toggle")
nnoremap <silent> <Plug>NERDCommenterToggle :call NERDComment("n", "Toggle")
xnoremap <silent> <Plug>NERDCommenterComment :call NERDComment("x", "Comment")
nnoremap <silent> <Plug>NERDCommenterComment :call NERDComment("n", "Comment")
nmap <silent> <Plug>RestoreWinPosn :call RestoreWinPosn()
nmap <silent> <Plug>SaveWinPosn :call SaveWinPosn()
nmap <SNR>18_WE <Plug>AlignMapsWrapperEnd
map <SNR>18_WS <Plug>AlignMapsWrapperStart
nnoremap <silent> <Plug>SurroundRepeat .
noremap <M-Right> 10>
noremap <M-Left> 10<
noremap <M-Down> 10-
noremap <M-Up> 10+
nnoremap <S-Right> l
nnoremap <S-Left> h
nnoremap <S-Down> j
nnoremap <S-Up> k
nnoremap <C-Down> gt
vnoremap <C-Down> :tabN
onoremap <C-Down> :tabN
nnoremap <C-Up> gT
noremap <F4> :tabe %
imap  ^^i
imap  $a
imap S <Plug>ISurround
imap s <Plug>Isurround
imap 	 <Plug>SuperTabForward
imap  
imap  <Plug>Isurround
cnoremap $v vs %:p:h
cnoremap $s split %:p:h
cnoremap $t tabe %:p:h
cnoremap $e e %:p:h
cnoremap $h e ~/
imap ,ihn :IHN
imap ,is :IHS
imap ,ih :IHS
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set background=dark
set backspace=indent,eol,start
set cindent
set expandtab
set fileencodings=ucs-bom,utf-8,default,latin1
set formatoptions=q
set helplang=en
set hlsearch
set ignorecase
set incsearch
set laststatus=2
set matchtime=1
set mouse=a
set printoptions=paper:letter
set report=0
set ruler
set runtimepath=~/.vim,~/.vim/bundle/Vundle.vim,~/.vim/bundle/ZoomWin,~/.vim/bundle/vim-surround,~/.vim/bundle/rainbow,~/.vim/bundle/a.vim,~/.vim/bundle/Align,~/.vim/bundle/taglist.vim,~/.vim/bundle/ack.vim,~/.vim/bundle/tagbar,~/.vim/bundle/vim-misc,~/.vim/bundle/vim-easytags,~/.vim/bundle/nerdcommenter,~/.vim/bundle/nerdtree,~/.vim/bundle/supertab,~/.vim/bundle/vimpdb,~/.vim/bundle/calendar.vim,~/.vim/bundle/vim-fugitive,~/.vim/bundle/fzf,~/.vim/bundle/TeTrIs.vim,~/.vim/bundle/vim-matrix-screensaver,~/.vim/bundle/darkburn,~/.vim/bundle/vim,~/.vim/bundle/zenburn,/var/lib/vim/addons,/usr/share/vim/vimfiles,/usr/share/vim/vim74,/usr/share/vim/vimfiles/after,/var/lib/vim/addons/after,~/.vim/after,~/.vim/bundle/Vundle.vim/after,~/.vim/bundle/ZoomWin/after,~/.vim/bundle/vim-surround/after,~/.vim/bundle/rainbow/after,~/.vim/bundle/a.vim/after,~/.vim/bundle/Align/after,~/.vim/bundle/taglist.vim/after,~/.vim/bundle/ack.vim/after,~/.vim/bundle/tagbar/after,~/.vim/bundle/vim-misc/after,~/.vim/bundle/vim-easytags/after,~/.vim/bundle/nerdcommenter/after,~/.vim/bundle/nerdtree/after,~/.vim/bundle/supertab/after,~/.vim/bundle/vimpdb/after,~/.vim/bundle/calendar.vim/after,~/.vim/bundle/vim-fugitive/after,~/.vim/bundle/fzf/after,~/.vim/bundle/TeTrIs.vim/after,~/.vim/bundle/vim-matrix-screensaver/after,~/.vim/bundle/darkburn/after,~/.vim/bundle/vim/after,~/.vim/bundle/zenburn/after
set scrolloff=4
set shiftwidth=4
set showcmd
set showmatch
set smartcase
set smartindent
set smarttab
set softtabstop=4
set nostartofline
set statusline=%{LastDir()}/%f%m\ %r\ %h\ %w%{GitBranch()}\ %=%l/%L:%c\ %015(%p%%%)
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc
set tabstop=4
set tags=./tags,./TAGS,tags,TAGS,~/.vimtags
set visualbell
set whichwrap=<,>,[,]
set wildignore=*.pyc
set wildmenu
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Desktop/workInProgress/networkofgraphs/process/PyNPB/src
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +0 hdp/mmsb.py
badd +0 ibp/ilfm_gs.py
argglobal
silent! argdel *
argadd hdp/mmsb.py
set stal=2
edit hdp/mmsb.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winheight=1 winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal cindent
setlocal cinkeys=0{,0},0),:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
set nofoldenable
setlocal nofoldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
set foldmethod=indent
setlocal foldmethod=indent
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=q
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=0
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(v:fname,'\\.','/','g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=!^F,o,O,<:>,0),0],0},=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=pydoc
set linebreak
setlocal linebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=pythoncomplete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=4
setlocal noshortname
setlocal smartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=4
setlocal tagcase=
setlocal tags=
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
let s:l = 656 - ((21 * winheight(0) + 21) / 42)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
656
normal! 0
tabedit ibp/ilfm_gs.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winheight=1 winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal cindent
setlocal cinkeys=0{,0},0),:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
set nofoldenable
setlocal nofoldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
set foldmethod=indent
setlocal foldmethod=indent
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=q
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=0
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(v:fname,'\\.','/','g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=!^F,o,O,<:>,0),0],0},=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=pydoc
set linebreak
setlocal linebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=pythoncomplete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=4
setlocal noshortname
setlocal smartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=4
setlocal tagcase=
setlocal tags=
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
let s:l = 128 - ((20 * winheight(0) + 21) / 42)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
128
normal! 05|
tabnext 2
set stal=1
if exists('s:wipebuf')
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToO
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :