import { Link } from "wouter";
import { DatabaseIcon } from "lucide-react";
import { FaGithub, FaTwitter, FaLinkedin } from "react-icons/fa";

const Footer = () => {
  return (
    <footer className="bg-white">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="xl:grid xl:grid-cols-3 xl:gap-8">
          <div className="space-y-8 xl:col-span-1">
            <div className="flex items-center">
              <DatabaseIcon className="h-8 w-8 text-primary-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">DataMeta AI</span>
            </div>
            <p className="text-gray-500 text-base">
              Enhancing AI research initiatives through high-quality metadata generation and dataset structuring.
            </p>
            <div className="flex space-x-6">
              <a href="#" className="text-gray-400 hover:text-gray-500">
                <FaGithub className="h-5 w-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-gray-500">
                <FaTwitter className="h-5 w-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-gray-500">
                <FaLinkedin className="h-5 w-5" />
              </a>
            </div>
          </div>
          <div className="mt-12 grid grid-cols-2 gap-8 xl:mt-0 xl:col-span-2">
            <div className="md:grid md:grid-cols-2 md:gap-8">
              <div>
                <h3 className="text-sm font-semibold text-gray-400 tracking-wider uppercase">
                  Product
                </h3>
                <ul role="list" className="mt-4 space-y-4">
                  <li>
                    <Link href="/#features">
                      <a className="text-base text-gray-500 hover:text-gray-900">
                        Features
                      </a>
                    </Link>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Pricing
                    </a>
                  </li>
                  <li>
                    <Link href="/api">
                      <a className="text-base text-gray-500 hover:text-gray-900">
                        API
                      </a>
                    </Link>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Integration
                    </a>
                  </li>
                </ul>
              </div>
              <div className="mt-12 md:mt-0">
                <h3 className="text-sm font-semibold text-gray-400 tracking-wider uppercase">
                  Resources
                </h3>
                <ul role="list" className="mt-4 space-y-4">
                  <li>
                    <Link href="/documentation">
                      <a className="text-base text-gray-500 hover:text-gray-900">
                        Documentation
                      </a>
                    </Link>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Guides
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Case Studies
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Webinars
                    </a>
                  </li>
                </ul>
              </div>
            </div>
            <div className="md:grid md:grid-cols-2 md:gap-8">
              <div>
                <h3 className="text-sm font-semibold text-gray-400 tracking-wider uppercase">
                  Company
                </h3>
                <ul role="list" className="mt-4 space-y-4">
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      About
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Blog
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Careers
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Contact
                    </a>
                  </li>
                </ul>
              </div>
              <div className="mt-12 md:mt-0">
                <h3 className="text-sm font-semibold text-gray-400 tracking-wider uppercase">
                  Legal
                </h3>
                <ul role="list" className="mt-4 space-y-4">
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Privacy
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Terms
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Cookie Policy
                    </a>
                  </li>
                  <li>
                    <a href="#" className="text-base text-gray-500 hover:text-gray-900">
                      Data Processing
                    </a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-12 border-t border-gray-200 pt-8">
          <p className="text-base text-gray-400 xl:text-center">
            &copy; {new Date().getFullYear()} DataMeta AI. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
