"------------------------------------------------------------------------------
"" Vim Plugin Manager
"------------------------------------------------------------------------------
call plug#begin('~/.local/share/nvim/plugged')

	Plug 'vim-airline/vim-airline'
	Plug 'vim-airline/vim-airline-themes'
	Plug 'altercation/vim-colors-solarized'

	Plug 'ctrlpvim/ctrlp.vim'
	Plug 'gcmt/taboo.vim'

  Plug 'tpope/vim-surround'
	Plug 'wesq3/vim-windowswap'
	Plug 'scrooloose/nerdcommenter'
  Plug 'easymotion/vim-easymotion'
  Plug 'terryma/vim-multiple-cursors'
	
	Plug 'majutsushi/tagbar'
	Plug 'shougo/vimshell.vim'
  Plug 'Shougo/vimproc.vim', {'do' : 'make'}
  Plug 'scrooloose/nerdtree', { 'on': 'NERDTreeToggle' }
	
	" auto complete "
	Plug 'valloric/youcompleteme'

	" Langs "
	Plug 'hynek/vim-python-pep8-indent'
	Plug 'sirver/ultisnips'

call plug#end()


"------------------------------------------------------------------------------
" VIM Config
"------------------------------------------------------------------------------
syntax on
filetype indent on
filetype plugin on

set	ruler
set colorcolumn=80
set t_Co=256 "tw=80

set number
set	showmode
set	cindent
set	autoindent

set	backspace=indent,eol,start

set	showmatch
set	incsearch
set	hlsearch
set	splitright
set	ignorecase

set smartindent
set expandtab
set sts=2
set tabstop=2
set shiftwidth=2

set	encoding=utf-8
set	fileencoding=utf-8
set	fileencodings=utf-8,korea,cp949
set	guifont=Monospace\ 15

set	autowrite		 	      "auto save when build
set	nobackup
set	noswapfile
set	clipboard=unnamed	  "윈도우즈 클립보드 사용


"------------------------------------------------------------------------------
" General Key Binding
"------------------------------------------------------------------------------
let mapleader='-'

nnoremap Q :qa<CR>
nnoremap Q! :q!<CR>
nnoremap <F9> :VimShellTab<CR>
nnoremap <Leader><F1>	:e ~/.config/nvim/init.vim<CR>
nnoremap <Leader><F9>	:VimShell -split ~/<CR>

nnoremap gn :bn<cr>
nnoremap gp :bp<cr>
nnoremap gd :bd<cr>

ca tn tabnew

"------------------------------------------------------------------------------
" Snipet
"------------------------------------------------------------------------------
"let g:UltiSnipsExpandTrigger="<tab>"
let g:UltiSnipsJumpForwardTrigger="<c-b>"
let g:UltiSnipsJumpBackwardTrigger="<c-z>"

"------------------------------------------------------------------------------
" Helper Key Binding
"------------------------------------------------------------------------------
" :vr vertical resize

"------------------------------------------------------------------------------
" General external library
"------------------------------------------------------------------------------
let ctagpath='/usr/bin/ctags'

"------------------------------------------------------------------------------
" Vim AutoCmd Handler
"------------------------------------------------------------------------------
autocmd VimEnter * wincmd p	"position cursor

"------------------------------------------------------------------------------
" NerdTree
"------------------------------------------------------------------------------
let g:NERDTreeWinSize=25
nnoremap <F5>	:NERDTreeToggle ~/<CR>
nnoremap <Leader><F5>1	:NERDTree ~/<CR>

let NERDTreeIgnore = ['\.ipynb$', '\.pyc$', '__pycache__$']

" USE <BAR> to multiple command
" nnoremap <F5>	:NERDTreeToggle<CR> <BAR> :NERDTree E:\<CR>
" autocmd VimEnter * NERDTreeToggle E:\

"------------------------------------------------------------------------------
" Airline
"------------------------------------------------------------------------------
set laststatus=2
let g:airline_theme='dark'

"------------------------------------------------------------------------------
" EasyMotion
"------------------------------------------------------------------------------
" s{char}{char} to move to {char}{char}
nmap s <Plug>(easymotion-overwin-f2)

"------------------------------------------------------------------------------
" CtrlP
"------------------------------------------------------------------------------
let g:ctrlp_match_window='top,order:ttb,min:1,max:10,results:10'
let g:ctrlp_working_path_mode='c'

"------------------------------------------------------------------------------
" TagBar
"------------------------------------------------------------------------------
let g:tagbar_ctags_bin=ctagpath
let g:tagbar_width=40
nmap <F8> :TagbarToggle<CR>

"------------------------------------------------------------------------------
" GUI Configuration
"------------------------------------------------------------------------------
" colorscheme molokai
hi Normal guibg=NONE ctermbg=NONE
