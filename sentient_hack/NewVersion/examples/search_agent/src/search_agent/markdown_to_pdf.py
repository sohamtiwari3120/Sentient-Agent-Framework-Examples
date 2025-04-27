from md2pdf.core import md2pdf
import argparse
import os

def convert_markdown_to_pdf(input_file: str, output_file: str = None):
    """
    Convert a Markdown file to PDF using md2pdf.
    
    Args:
        input_file (str): Path to the input Markdown file
        output_file (str, optional): Path to save the output PDF file. 
                                   If not provided, will use same name as input with .pdf extension
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file is None:
        # If no output file specified, use same name as input but with .pdf extension
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.pdf"
    
    print(f"Converting {input_file} to {output_file}")
    md2pdf(
        pdf_file_path=output_file,
        md_file_path=input_file,
        css_file_path=None,  # Optional CSS styling
        base_url=None
    )
    print(f"Successfully converted to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert Markdown files to PDF')
    parser.add_argument('input_file', help='Input Markdown file')
    parser.add_argument('--output', help='Output PDF file (optional)')
    
    args = parser.parse_args()
    
    try:
        convert_markdown_to_pdf(args.input_file, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()