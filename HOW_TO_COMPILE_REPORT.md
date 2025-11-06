# How to Compile the Formal Report

## 🎯 Quick Answer: Where to Run Commands

You need to run the LaTeX compilation commands in your **terminal/command prompt**, in the directory where `formal_report.tex` is located.

---

## 📍 Step 1: Navigate to the Correct Directory

### On Windows (PowerShell or Command Prompt):
```powershell
# Navigate to your project folder
cd "C:\Users\AKANKSHA BHARTI\OneDrive\Documents\TerraViT\TerraViT-main"
```

### On Mac/Linux (Terminal):
```bash
cd ~/Documents/TerraViT/TerraViT-main
# OR
cd /path/to/TerraViT/TerraViT-main
```

**Verify you're in the right place:**
```bash
# List files - you should see formal_report.tex
dir          # Windows
ls           # Mac/Linux
```

---

## 🛠️ Step 2: Choose Your Compilation Method

### **Option A: Overleaf (EASIEST - Recommended if you don't have LaTeX installed)**

1. **Go to**: https://www.overleaf.com
2. **Sign up/Login** (free account)
3. **Create New Project** → "Upload Project"
4. **Upload these files**:
   - `formal_report.tex`
   - `references.bib`
   - Any ACL style files (Overleaf may have them pre-installed)
5. **Click "Recompile"** button
6. **View PDF** - page count will be visible in the PDF viewer

**Advantages:**
- ✅ No installation needed
- ✅ Works in web browser
- ✅ Automatic compilation
- ✅ Easy to share/collaborate

---

### **Option B: Local LaTeX Installation**

#### **Prerequisites:**
You need LaTeX installed on your computer:

**Windows:**
- Install **MiKTeX** (https://miktex.org/download) OR
- Install **TeX Live** (https://www.tug.org/texlive/)

**Mac:**
- Install **MacTeX** (https://www.tug.org/mactex/)

**Linux:**
```bash
sudo apt-get install texlive-full  # Ubuntu/Debian
```

#### **Compilation Steps:**

Once LaTeX is installed, open terminal/command prompt in your project directory and run:

```bash
# Step 1: Compile LaTeX (first pass)
pdflatex formal_report.tex

# Step 2: Process bibliography
bibtex formal_report

# Step 3: Compile again (to include citations)
pdflatex formal_report.tex

# Step 4: Final compile (to resolve all references)
pdflatex formal_report.tex
```

**Output:**
- A file `formal_report.pdf` will be created
- Open it to check the page count

---

## 📄 Step 3: Check Page Count

After compilation:

1. **Open** `formal_report.pdf`
2. **Count pages** (excluding the References page at the end)
3. **Target**: Should be **4-5 pages** of content

**Example:**
- If PDF has 6 pages total:
  - Pages 1-5 = Content ✅ (5 pages - meets requirement)
  - Page 6 = References (excluded from count)

---

## 🔧 Troubleshooting

### Problem: "pdflatex: command not found"
**Solution**: LaTeX is not installed. Use **Option A (Overleaf)** instead.

### Problem: "ACL2023.sty not found"
**Solution**: 
- Download ACL style files from: https://github.com/acl-org/acl-style-files
- Place `ACL2023.sty` in the same directory as `formal_report.tex`
- OR use Overleaf (has ACL templates pre-installed)

### Problem: Bibliography not compiling
**Solution**: Make sure `references.bib` is in the same directory as `formal_report.tex`

### Problem: Compilation errors
**Solution**: 
- Check for typos in LaTeX syntax
- Ensure all `\cite{}` commands have corresponding entries in `references.bib`
- Use Overleaf for better error messages

---

## ✅ Quick Checklist

- [ ] Navigated to project directory (where `formal_report.tex` is)
- [ ] Chosen compilation method (Overleaf or Local)
- [ ] Compiled successfully (no errors)
- [ ] Opened PDF and counted pages (excluding references)
- [ ] Verified page count is 4-5 pages

---

## 🎯 Recommended Approach

**For most users**: Use **Overleaf** (Option A)
- No installation required
- Works immediately
- Easy to use
- Professional output

**For advanced users**: Use **Local LaTeX** (Option B)
- Faster compilation
- More control
- Works offline

---

## 📝 Next Steps After Compilation

Once you've verified the page count:

1. ✅ If **4-5 pages**: Report is ready!
2. ⚠️ If **< 4 pages**: Add more content (especially in Analysis section)
3. ⚠️ If **> 5 pages**: Condense some sections

---


